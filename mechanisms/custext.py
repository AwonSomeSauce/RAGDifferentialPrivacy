import os
import json
import logging
from collections import defaultdict
import numpy as np
from tqdm import trange
from sklearn.metrics.pairwise import euclidean_distances
from .base_mechanism import BaseMechanism

class CusText(BaseMechanism):
    """Class for sanitizing text by using CusTextPlus mechanism"""
    PROBABILITY_MAPPINGS_PATH = "./word_mappings/probability_mappings.txt"
    SIMILAR_WORD_MAPPINGS_PATH = "./word_mappings/similar_word_mappings.txt"

    def __init__(self, word_embedding, word_embedding_path, epsilon, top_k, detector):
        super().__init__(word_embedding, word_embedding_path, epsilon)
        self.top_k = top_k
        self.detector = detector

    def sanitize(self, dataset):
        """Sanitize dataset"""
        return self._transform_sentences(dataset)

    def _transform_sentences(self, df):
        """Transform sentences in the dataframe"""
        # Lowercase the entire 'sentence' column if using GloVe embeddings
        if self.word_embedding == 'glove':
            df['sentence'] = df['sentence'].str.lower()

        probability_mappings, similar_word_cache = self._load_or_generate_word_mappings(df)
        
        logging.info(" Santizing dataset using CusText")

        sensitive_words = self.detector.detect(similar_word_cache.keys())

        transformed_sentences = [
            " ".join([self._get_substituted_word(word, similar_word_cache, probability_mappings, sensitive_words) 
                      for word in sentence.split()]) 
            for sentence in df.sentence]
        
        df_copy = df.copy()
        df_copy.sentence = transformed_sentences

        return df_copy

    def _load_or_generate_word_mappings(self, df):
        """Load word mappings from files or generate if not available"""
        if os.path.exists(self.PROBABILITY_MAPPINGS_PATH) and os.path.exists(self.SIMILAR_WORD_MAPPINGS_PATH):
            logging.info(" Found word mappings in %s, %s. Making use of them.", self.PROBABILITY_MAPPINGS_PATH, self.SIMILAR_WORD_MAPPINGS_PATH)

            with open(self.PROBABILITY_MAPPINGS_PATH, "r", encoding="utf-8") as file:
                probability_mappings = json.load(file)
            with open(self.SIMILAR_WORD_MAPPINGS_PATH, "r", encoding="utf-8") as file:
                similar_word_cache = json.load(file)
            return probability_mappings, similar_word_cache

        return self._generate_word_mappings(df)

    def _get_substituted_word(self, word, similar_word_cache, probability_mappings, sensitive_words):
        """Return a substituted word based on mappings or the original word"""
        if word not in sensitive_words:
            return word

        if word not in similar_word_cache:
            return str(round(float(word))+np.random.randint(1000)) if word.isdigit() else word

        substitution_probabilities = probability_mappings[word]
        return np.random.choice(similar_word_cache[word], 1, p=substitution_probabilities)[0]

    def _generate_word_mappings(self, df):
        """Generate word substitution mappings based on embeddings and save to files"""
        logging.info(" Generating word mappings and saving them in %s and %s", self.PROBABILITY_MAPPINGS_PATH, self.SIMILAR_WORD_MAPPINGS_PATH)

        word_frequencies = self._compute_word_frequencies(df)
        embeddings, word_to_index, index_to_word = self._load_word_embeddings()

        substituted_word_dict = defaultdict(str)
        similar_word_dict = defaultdict(list)
        probability_dict = defaultdict(list)

        for index in trange(len(word_frequencies)):
            word = word_frequencies[index]
            if word in word_to_index and word not in substituted_word_dict:
                similar_indices = euclidean_distances(embeddings[word_to_index[word]].reshape(1, -1), embeddings)[0].argsort()[:self.top_k]
                similar_words = [index_to_word[idx] for idx in similar_indices]
                similar_embeddings = np.array([embeddings[idx] for idx in similar_indices])

                for similar_word in similar_words:
                    if similar_word not in substituted_word_dict:
                        substituted_word_dict[similar_word] = word
                        similarity_distances = euclidean_distances(embeddings[word_to_index[similar_word]].reshape(1, -1), similar_embeddings)[0]
                        normalized_distances = self._normalize_distances(similarity_distances)
                        adjusted_probs = [np.exp(self.epsilon * distance / 2) for distance in normalized_distances]
                        total_prob = sum(adjusted_probs)
                        probabilities = [prob / total_prob for prob in adjusted_probs]
                        
                        probability_dict[similar_word] = probabilities
                        similar_word_dict[similar_word] = similar_words

        self._save_to_file(self.PROBABILITY_MAPPINGS_PATH, probability_dict)
        self._save_to_file(self.SIMILAR_WORD_MAPPINGS_PATH, similar_word_dict)

        return probability_dict, similar_word_dict
