import os
import json
from collections import defaultdict
import numpy as np
from tqdm import trange
from sklearn.metrics.pairwise import euclidean_distances
from .base_mechanism import BaseMechanism

class CusText(BaseMechanism):
    def __init__(self, word_embedding, word_embedding_path, epsilon, top_k):
        super().__init__(word_embedding, word_embedding_path, epsilon)
        self.top_k = top_k

    def sanitize(self, dataset):
        return self.transform_sentences(dataset)

    def transform_sentences(self, df):
        probability_mappings, similar_word_cache = self._load_or_generate_word_mappings(df)
        
        new_df = df.copy()
        transformed_dataset = []

        for sentence in trange(df.shape[0]):
            words = df.sentence.iloc[sentence].split()
            new_words = [self._get_substituted_word(word, similar_word_cache, probability_mappings) for word in words]
            transformed_dataset.append(" ".join(new_words))

        new_df.sentence = transformed_dataset

        return new_df

    def _load_or_generate_word_mappings(self, df):
        probability_mappings_path = "./word_mappings/probability_mappings.txt"
        similar_word_cache_path = "./word_mappings/similar_word_mappings.txt"

        if os.path.exists(probability_mappings_path) and os.path.exists(similar_word_cache_path):
            with open(probability_mappings_path, "r") as file:
                probability_mappings = json.load(file)
            with open(similar_word_cache_path, "r") as file:
                similar_word_cache = json.load(file)
            return probability_mappings, similar_word_cache
        
        return self.generate_word_mappings(df)

    def _get_substituted_word(self, word, similar_word_cache, probability_mappings):
        if word not in similar_word_cache:
            return str(int(word) + np.random.randint(1000)) if word.isdigit() else word
        
        substitution_probabilities = probability_mappings[word]
        return np.random.choice(similar_word_cache[word], p=substitution_probabilities)

    def generate_word_mappings(self, df):
        word_frequencies = self._compute_word_frequencies(df)

        embeddings, word_to_index, index_to_word = self._load_word_embeddings()

        word_mappings = defaultdict(str)
        similar_word_cache = defaultdict(list)
        probability_mappings = defaultdict(list)

        for word in trange(len(word_frequencies)):
            if word in word_to_index and word not in word_mappings:
                closest_indices = self._get_closest_word_indices(embeddings, word_to_index, word)
                similar_words = [index_to_word[idx] for idx in closest_indices]
                similar_word_embeddings = np.array([embeddings[idx] for idx in closest_indices])

                for similar_word in similar_words:
                    if similar_word not in word_mappings:
                        word_mappings[similar_word] = word
                        distances = euclidean_distances(
                            embeddings[word_to_index[similar_word]].reshape(1, -1),
                            similar_word_embeddings,
                        )[0]
                        normalized_distances = self._normalize_distances(distances)
                        probabilities = [np.exp(self.epsilon * dist / 2) for dist in normalized_distances]
                        normalized_probabilities = [prob / sum(probabilities) for prob in probabilities]

                        probability_mappings[similar_word] = normalized_probabilities
                        similar_word_cache[similar_word] = similar_words

        self._save_to_file("./word_mappings/probability_mappings.txt", probability_mappings)
        self._save_to_file("./word_mappings/similar_word_mappings.txt", similar_word_cache)

        return similar_word_cache, probability_mappings
