import os
import json
import logging
import numpy as np
from collections import defaultdict
from tqdm import trange
from spacy.lang.en import English
from sklearn.metrics.pairwise import euclidean_distances
from .base_mechanism import BaseMechanism


class DisText(BaseMechanism):
    def __init__(
        self, word_embedding, word_embedding_path, epsilon, distance, detector
    ):
        super().__init__(word_embedding, word_embedding_path, epsilon)
        self.distance = distance
        self.detector = detector
        self.PROBABILITY_MAPPINGS_PATH = f"./word_mappings/distext/probability_mappings_{self.epsilon}_and_{distance}.txt"
        self.SIMILAR_WORD_MAPPINGS_PATH = f"./word_mappings/distext/similar_word_mappings_{self.epsilon}_and_{distance}.txt"
        self.MISSING_WORDS_PATH = "missing_words_distext.csv"

    def sanitize(self, dataset):
        """Sanitize dataset"""
        return self._transform_sentences(dataset)

    def _transform_sentences(self, df):
        """Transform sentences in the dataframe"""

        probability_mappings, similar_word_cache = self._load_or_generate_word_mappings(
            df
        )

        logging.info(" Santizing dataset using CusText")

        sensitive_words = self.detector.detect(similar_word_cache.keys())
        tokenizer = English()

        sanitized_sentences = []
        sentence_sensitive_words = []
        sentence_missing_words = []
        for sentence in df["sentence"]:
            tokens = [token.text for token in tokenizer(sentence)]
            sanitized = " ".join(
                [
                    self._get_substituted_word(
                        word, similar_word_cache, probability_mappings, sensitive_words
                    )
                    for word in tokens
                ]
            )
            sens_words = [
                word
                for word in sensitive_words
                if self.match_whole_word(word, sentence)
            ]
            miss_words = [
                word
                for word in self.missing_words
                if self.match_whole_word(word, sentence)
            ]
            sanitized_sentences.append(sanitized)
            sentence_sensitive_words.append(sens_words)
            sentence_missing_words.append(miss_words)

        sanitized_df = df.copy()
        sanitized_df["sanitized sentence"] = sanitized_sentences
        sanitized_df["sensitive words"] = sentence_sensitive_words
        sanitized_df["missing words"] = sentence_missing_words
        if not os.path.exists(self.MISSING_WORDS_PATH):
            self._save_missing_words_to_csv(self.MISSING_WORDS_PATH)
        return sanitized_df

    def _load_or_generate_word_mappings(self, df):
        """Load word mappings from files or generate if not available"""
        if os.path.exists(self.PROBABILITY_MAPPINGS_PATH) and os.path.exists(
            self.SIMILAR_WORD_MAPPINGS_PATH
        ):
            logging.info(
                " Found word mappings in %s, %s. Making use of them.",
                self.PROBABILITY_MAPPINGS_PATH,
                self.SIMILAR_WORD_MAPPINGS_PATH,
            )

            with open(self.PROBABILITY_MAPPINGS_PATH, "r", encoding="utf-8") as file:
                probability_mappings = json.load(file)
            with open(self.SIMILAR_WORD_MAPPINGS_PATH, "r", encoding="utf-8") as file:
                similar_word_cache = json.load(file)
            return probability_mappings, similar_word_cache

        return self._generate_word_mappings(df)

    def _get_substituted_word(
        self, word, similar_word_cache, probability_mappings, sensitive_words
    ):
        """Return a substituted word based on mappings or the original word"""
        if word not in similar_word_cache:
            self.missing_words.append(word)

            # If the word is not sensitive, return as is
            if word not in sensitive_words:
                return word

            # If the word is a digit, modify it
            if word.isdigit():
                return str(int(word) + np.random.randint(1000))
            else:
                return word

        if word not in sensitive_words:
            return word

        # If the word is sensitive and in the cache, use the substitution probabilities
        substitution_probabilities = probability_mappings[word]
        return np.random.choice(
            similar_word_cache[word], 1, p=substitution_probabilities
        )[0]

    def _generate_word_mappings(self, df):
        """Generate word substitution mappings based on embeddings and save to files"""
        logging.info(
            " Generating word mappings and saving them in %s and %s",
            self.PROBABILITY_MAPPINGS_PATH,
            self.SIMILAR_WORD_MAPPINGS_PATH,
        )

        # print("CALCULATING WORD FREQUENCIES")
        word_frequencies = self._compute_word_frequencies(df)
        embeddings, word_to_index, index_to_word = self._load_word_embeddings()

        substituted_word_dict = defaultdict(str)
        similar_word_dict = defaultdict(list)
        probability_dict = defaultdict(list)

        # print("WORD FREQUENCIES LENGTH: ", len(word_frequencies))
        for index in trange(len(word_frequencies)):
            word = word_frequencies[index]
            if word in word_to_index and word not in substituted_word_dict:
                # Calculate Euclidean distances
                distances = euclidean_distances(
                    embeddings[word_to_index[word]].reshape(1, -1), embeddings
                )[0]

                # Round distances to three decimal places
                rounded_distances = np.round(distances, 3)

                # Filter indices based on the distance threshold
                valid_indices = np.where(rounded_distances <= self.distance)[0]

                # Sort these indices based on their distances and select the top_k
                similar_indices = valid_indices[np.argsort(distances[valid_indices])]
                similar_words = [index_to_word[idx] for idx in similar_indices]
                similar_embeddings = np.array(
                    [embeddings[idx] for idx in similar_indices]
                )

                for similar_word in similar_words:
                    if (
                        similar_word in word_frequencies
                        and similar_word in word_to_index
                        and similar_word not in substituted_word_dict
                    ):
                        substituted_word_dict[similar_word] = word
                        similarity_distances = euclidean_distances(
                            embeddings[word_to_index[similar_word]].reshape(1, -1),
                            similar_embeddings,
                        )[0]
                        normalized_distances = self._normalize_distances(
                            similarity_distances
                        )
                        adjusted_probs = [
                            np.exp(self.epsilon * distance / 2)
                            for distance in normalized_distances
                        ]
                        total_prob = sum(adjusted_probs)
                        probabilities = [prob / total_prob for prob in adjusted_probs]

                        probability_dict[similar_word] = probabilities
                        similar_word_dict[similar_word] = similar_words

        self._save_to_file(self.PROBABILITY_MAPPINGS_PATH, probability_dict)
        self._save_to_file(self.SIMILAR_WORD_MAPPINGS_PATH, similar_word_dict)

        return probability_dict, similar_word_dict
