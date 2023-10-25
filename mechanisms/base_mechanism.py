import os
import json
import unicodedata
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

class BaseMechanism:
    def __init__(self, word_embedding, word_embedding_path, epsilon):
        self.word_embedding = word_embedding
        self.word_embedding_path = word_embedding_path
        self.epsilon = epsilon

    def sanitize(self, dataset):
        raise NotImplementedError

    def _load_word_embeddings(self, mechanism='SanText'):
        if (mechanism == 'CusText'):
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
        else:
            word_to_id, sensitive_word_to_id = {}, {}
            general_word_embeddings, sensitive_word_embeddings = [], []

            num_lines = sum(1 for _ in open(self.word_embedding_path))

            with open(self.word_embedding_path) as file:
                # Handle potential header in word embeddings file
                if len(file.readline().split()) != 2:
                    file.seek(0)

                for row in tqdm(file, total=num_lines - 1):
                    content = row.rstrip().split(' ')
                    current_word = content[0]
                    embedding = [float(i) for i in content[1:]]

                    if current_word in vocab and current_word not in word_to_id:
                        word_to_id[current_word] = len(general_word_embeddings)
                        general_word_embeddings.append(embedding)
                        
                        if current_word in self.sensitive_words_to_id:
                            sensitive_word_to_id[current_word] = len(sensitive_word_embeddings)
                            sensitive_word_embeddings.append(embedding)

            return np.array(general_word_embeddings), np.array(sensitive_word_embeddings), word_to_id, sensitive_word_to_id

    def _compute_word_frequencies(self, df):
        corpus = " ".join(df.sentence)
        word_frequencies = [
            word[0]
            for word in Counter(corpus.split()).most_common()
            if word[0] not in stop_words
        ]
        return word_frequencies

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
