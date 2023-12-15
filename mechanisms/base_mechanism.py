import os
import csv
import json
import re
import logging
from spacy.lang.en import English
from collections import Counter
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)


class BaseMechanism:
    """Base class for SanText and CusText"""

    def __init__(self, word_embedding, word_embedding_path, epsilon):
        self.word_embedding = word_embedding
        self.word_embedding_path = word_embedding_path
        self.epsilon = epsilon
        self.missing_words = []

    def sanitize(self, dataset):
        """Sanitize the given dataset"""
        raise NotImplementedError

    def _load_word_embeddings(self):
        """Load word embeddings from a file"""
        embeddings = []
        index_to_word = []
        word_to_index = {}

        with open(self.word_embedding_path, "r", encoding="utf-8") as file:
            if not self._has_header(file):
                file.seek(0)

            for row in tqdm(file):
                content = row.rstrip().split(" ")
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
        tokenizer = English()
        vocab = Counter()
        for text in df["sentence"]:
            tokenized_text = [token.text for token in tokenizer(text) if token.is_alpha]
            vocab.update(tokenized_text)

        return [word[0] for word in vocab.most_common()]

    def _normalize_distances(self, distances):
        """Normalize the given distances"""
        distance_range = max(distances) - min(distances)
        min_distance = min(distances)
        return [-(dist - min_distance) / distance_range for dist in distances]

    def match_whole_word(self, text, word):
        pattern = r"\b{}\b".format(re.escape(word))
        return re.search(pattern, text) is not None

    def _save_missing_words_to_csv(self, file_path):
        """Save the list of missing words to a CSV file"""
        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["word"])  # header row
            for word in set(self.missing_words):  # writing unique words
                csvwriter.writerow([word])

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
