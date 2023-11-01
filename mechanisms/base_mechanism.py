import os
import json
import logging
from collections import Counter
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)

class BaseMechanism:
    """Base class for SanText and CusText"""
    def __init__(self, word_embedding, word_embedding_path, epsilon):
        self.word_embedding = word_embedding
        self.word_embedding_path = word_embedding_path
        self.epsilon = epsilon

    def sanitize(self, dataset):
        """Sanitize the given dataset"""
        raise NotImplementedError

    def _load_word_embeddings(self):
        """Load word embeddings from a file"""
        embeddings = []
        index_to_word = []
        word_to_index = {}

        with open(self.word_embedding_path, "r",  encoding="utf-8") as file:
            if not self._has_header(file):
                file.seek(0)
            num_lines = sum(1 for _ in file)
            file.seek(0)

            for row in tqdm(file, total=num_lines):
                content = row.rstrip().split(' ')
                word, vector = content[0], list(map(float, content[1:]))
                index_to_word.append(word)
                word_to_index[word] = len(index_to_word) - 1
                embeddings.append(vector)

        return np.asarray(embeddings), word_to_index, np.asarray(index_to_word)

    def _has_header(self, file):
        """Check if the embeddings file has a header"""
        return len(file.readline().split()) == 2

    def _compute_word_frequencies(self, df):
        """Compute word frequencies from the dataframe"""
        corpus = " ".join(df.sentence)
        all_words = Counter(corpus.split()).most_common()

        return [word[0] for word in all_words]

    def _normalize_distances(self, distances):
        """Normalize the given distances"""
        distance_range = max(distances) - min(distances)
        min_distance = min(distances)
        return [-(dist - min_distance) / distance_range for dist in distances]

    def _save_to_file(self, file_path, data):
        """Save the given data to a file"""
        self._ensure_directory_exists(file_path)

        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except IOError as e:
            raise IOError(f"Error writing to {file_path}. Reason: {e}") from e

    @staticmethod
    def _ensure_directory_exists(file_path):
        """Ensure the directory for the given file path exists"""
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
