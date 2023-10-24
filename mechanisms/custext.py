import os
from collections import Counter, defaultdict
import numpy as np
import nltk
import json
from tqdm import trange
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from .base_mechanism import BaseMechanism


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


class CusText(BaseMechanism):
    def __init__(self, word_embedding, word_embedding_path, epsilon, top_k):
        super().__init__(word_embedding, word_embedding_path, epsilon)
        self.top_k = top_k

    def generate_word_mappings(self, df):
        train_corpus = " ".join(df.sentence)
        word_frequencies = [
            word[0]
            for word in Counter(train_corpus.split()).most_common()
            if word[0] not in stop_words
        ]

        embeddings, word_to_index, index_to_word = self._load_word_embeddings()

        word_mappings = defaultdict(str)
        similar_word_cache = defaultdict(list)
        probability_mappings = defaultdict(list)

        for word in trange(len(word_frequencies)):
            if word in word_to_index:
                if word not in word_mappings:
                    closest_indices = self._get_closest_word_indices(
                        embeddings, word_to_index, word
                    )
                    similar_words = [index_to_word[idx] for idx in closest_indices]
                    similar_word_embeddings = np.array(
                        [embeddings[idx] for idx in closest_indices]
                    )

                    for similar_word in similar_words:
                        if similar_word not in word_mappings:
                            word_mappings[similar_word] = word
                            distances = euclidean_distances(
                                embeddings[word_to_index[similar_word]].reshape(1, -1),
                                similar_word_embeddings,
                            )[0]
                            normalized_distances = self._normalize_distances(distances)
                            probabilities = [
                                np.exp(self.epsilon * dist / 2)
                                for dist in normalized_distances
                            ]
                            normalized_probabilities = [
                                prob / sum(probabilities) for prob in probabilities
                            ]

                            probability_mappings[
                                similar_word
                            ] = normalized_probabilities
                            similar_word_cache[similar_word] = similar_words

        self._save_to_file(
            "./word_mappings/probability_mappings.txt", probability_mappings
        )
        self._save_to_file(
            "./word_mappings/similar_word_mappings.txt", similar_word_cache
        )

        return similar_word_cache, probability_mappings

    def transform_sentences(self, df):
        probability_mappings_path = "./word_mappings/probability_mappings.txt"
        similar_word_cache_path = "./word_mappings/similar_word_mappings.txt"

        if os.path.exists(probability_mappings_path) and os.path.exists(
            similar_word_cache_path
        ):
            with open(probability_mappings_path, "r") as file:
                probability_mappings = json.load(file)
            with open(similar_word_cache_path, "r") as file:
                similar_word_cache = json.load(file)
        else:
            similar_word_cache, probability_mappings = self.generate_word_mappings(df)

        new_df = df.copy()
        transformed_dataset = []

        for idx in trange(len(df.sentence)):
            sentence = df.sentence.iloc[idx]
            words = sentence.split()
            new_words = []

            for word in words:
                if word not in similar_word_cache:
                    if word.isdigit():
                        word = str(int(word) + np.random.randint(1000))
                    new_words.append(word)
                else:
                    substitution_probabilities = probability_mappings[word]
                    substituted_word = np.random.choice(
                        similar_word_cache[word], 1, p=substitution_probabilities
                    )[0]
                    new_words.append(substituted_word)

            transformed_dataset.append(" ".join(new_words))

        new_df.sentence = transformed_dataset

        return new_df

    def sanitize(self, dataset):
        return self.transform_sentences(dataset)

    def _load_word_embeddings(self):
        embeddings = []
        index_to_word = []
        word_to_index = {}

        with open(self.word_embedding_path, "r") as file:
            for row in file:
                content = row.rstrip().split(" ")
                if len(content) > 2:  # To skip potential count/dimension rows
                    word = content[0]
                    vector = list(map(float, content[1:]))
                    index_to_word.append(word)
                    word_to_index[word] = len(index_to_word) - 1
                    embeddings.append(vector)

        return np.asarray(embeddings), word_to_index, np.asarray(index_to_word)

    def _get_closest_word_indices(self, embeddings, word_to_index, word):
        distances = euclidean_distances(
            embeddings[word_to_index[word]].reshape(1, -1), embeddings
        )[0]
        return distances.argsort()[: self.top_k]

    def _normalize_distances(self, distances):
        distance_range = max(distances) - min(distances)
        return [-(dist - min(distances)) / distance_range for dist in distances]

    def _save_to_file(self, file_path, data):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        try:
            with open(file_path, "w") as file:
                file.write(json.dumps(data, ensure_ascii=False, indent=4))
        except IOError:
            print(f"Error writing to {file_path}")
