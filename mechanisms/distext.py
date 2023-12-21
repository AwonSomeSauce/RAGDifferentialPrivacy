import logging
import numpy as np
from collections import defaultdict
from tqdm import trange
from sklearn.metrics.pairwise import euclidean_distances
from .custext import CusText


class DisText(CusText):
    def __init__(self, word_embedding, word_embedding_path, epsilon, top_k, detector):
        super().__init__(word_embedding, word_embedding_path, epsilon, top_k, detector)
        self.PROBABILITY_MAPPINGS_PATH = (
            f"./word_mappings/distext/probability_mappings_{self.epsilon}.txt"
        )
        self.SIMILAR_WORD_MAPPINGS_PATH = (
            f"./word_mappings/distext/similar_word_mappings_{self.epsilon}.txt"
        )

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
                valid_indices = np.where(rounded_distances <= 5.500)[0]

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
