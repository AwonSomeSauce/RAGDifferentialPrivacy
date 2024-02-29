from collections import Counter, defaultdict
import random
import re
import pdb
import numpy as np
from tqdm import tqdm
import scipy.spatial.distance as sp_dist
from spacy.lang.en import English
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from .base_mechanism import BaseMechanism

# Set the seed for NumPy's random number generator.
np.random.seed(42)


class kText(BaseMechanism):
    """Class for sanitizing text by using SanTextPlus mechanism"""

    MISSING_WORDS_PATH = "missing_words_santext.csv"

    def __init__(
        self,
        word_embedding,
        word_embedding_path,
        epsilon,
        p,
        detector,
        top_k,
        distance_metric,
    ):
        super().__init__(word_embedding, word_embedding_path, epsilon)
        self.p = p
        self.detector = detector
        self.top_k = top_k
        self.distance_metric = distance_metric

    def build_vocab_from_dataset(self, df):
        """Build vocabulary from dataset"""
        tokenizer = English()
        vocab = Counter()
        for text in df["sentence"]:
            tokenized_text = [
                token.text
                for token in tokenizer(text)
                if (token.is_alpha or token.is_digit)
            ]
            vocab.update(tokenized_text)
        return vocab

    def process_word_embeddings(self, vocab):
        """Process word general_embeddings and return arrays and dictionaries for words in the vocabulary"""
        word_to_id = {}
        general_word_embeddings = []
        num_lines = sum(1 for _ in open(self.word_embedding_path, encoding="utf-8"))

        with open(self.word_embedding_path, encoding="utf-8") as file:
            if not self._has_header(file):
                file.seek(0)
            num_lines = sum(1 for _ in file)
            file.seek(0)

            for row in tqdm(file, total=num_lines - 1):
                word, embedding = self._parse_embedding_row(row)
                self._process_word_embedding(
                    word,
                    embedding,
                    vocab,
                    word_to_id,
                    general_word_embeddings,
                )

        return (np.asarray(general_word_embeddings), word_to_id)

    def sanitize(self, dataset):
        """Sanitize dataset"""
        return self._transform_sentences(dataset)

    def get_sensitivity(self, words, word_to_id, general_embeddings):
        max_distance_dict = defaultdict(float)

        for word in words:
            if word in word_to_id and word not in max_distance_dict:
                metric_parameters = {}
                if self.distance_metric == "minkowski":
                    metric_parameters["p"] = 3
                similar_indices = sp_dist.cdist(
                    general_embeddings[word_to_id[word]].reshape(1, -1),
                    general_embeddings,
                    metric=self.distance_metric,
                    **metric_parameters
                )[0].argsort()[: self.top_k]
                max_distance_index = similar_indices[-1]
                max_distance = sp_dist.cdist(
                    general_embeddings[word_to_id[word]].reshape(1, -1),
                    general_embeddings[max_distance_index].reshape(1, -1),
                    metric=self.distance_metric,
                    **metric_parameters
                )
                max_distance_dict[word] = max_distance

        # Finding the word with the greatest maximum distance
        word_with_greatest_dist = max(max_distance_dict, key=max_distance_dict.get)
        print("The word with furthest neighbours is: ", word_with_greatest_dist)
        return max_distance_dict[word_with_greatest_dist]

    def _transform_sentences(self, df):
        """Transform sentences in the dataframe"""

        vocab = self.build_vocab_from_dataset(df)
        words = [key for key, _ in vocab.most_common()]
        sensitive_words = self.detector.detect(vocab)
        processed_data = self.process_word_embeddings(vocab)
        (general_embeddings, word_to_id) = processed_data
        id_to_word = {v: k for k, v in word_to_id.items()}
        sensitivity = self.get_sensitivity(words, word_to_id, general_embeddings)

        sanitized_sentences = []

        for sentence in tqdm(
            df["sentence"],
            desc="Sanitizing Sentences",
            unit="sentence",
            total=len(df["sentence"]),
        ):
            sanitized = self._sanitize_sentence(
                sentence,
                general_embeddings,
                word_to_id,
                id_to_word,
                words,
                sensitive_words,
                sensitivity,
            )
            sanitized_sentences.append(sanitized)

        sanitized_df = df.copy()
        sanitized_df["sanitized sentence"] = sanitized_sentences

        return sanitized_df

    def _sanitize_sentence(
        self,
        sentence,
        general_embeddings,
        word_to_id,
        id_to_word,
        words,
        sensitive_words,
        sensitivity,
    ):
        """Sanitize individual sentence"""
        tokenizer = English()
        tokens = [
            token.text
            for token in tokenizer(sentence)
            if (token.is_alpha or token.is_digit)
        ]
        sanitized_tokens = []
        for word in tokens:
            sanitized_tokens.append(
                self._get_word_substitute_or_original(
                    word,
                    general_embeddings,
                    word_to_id,
                    id_to_word,
                    words,
                    sensitive_words,
                    sensitivity,
                )
            )
        return " ".join(sanitized_tokens)

    def _get_word_substitute_or_original(
        self,
        word,
        general_embeddings,
        word_to_id,
        id_to_word,
        words,
        sensitive_words,
        sensitivity,
    ):
        """Get substitute for word or return original if not sanitized"""
        if re.match(r"^\d+$", word):
            return self._generate_random_number_string(len(word))

        if word in word_to_id:
            if word in sensitive_words:
                return self._get_substitute_word(
                    word,
                    general_embeddings,
                    word_to_id,
                    id_to_word,
                    sensitivity,
                )
            else:
                return word

        return self._handle_out_of_vocab_word(words)

    def _get_substitute_word(
        self,
        word,
        general_embeddings,
        word_to_id,
        id_to_word,
        sensitivity,
    ):
        """Retrieve a substitute word"""
        # similar_words = similar_word_cache[word]

        # Filter embeddings and word_to_id based on similar_words
        # cache_word_ids = [word_to_id[w] for w in similar_words if w in word_to_id]
        # cache_embeddings = general_embeddings[cache_word_ids]
        distances = euclidean_distances(
            general_embeddings[word_to_id[word]].reshape(1, -1),
            general_embeddings,
        )[0]
        sim_matrix = -distances
        prob_matrix = softmax(self.epsilon * sim_matrix / (2 * sensitivity))
        substitute_idx = np.random.choice(len(prob_matrix), 1, p=prob_matrix)
        return id_to_word[substitute_idx[0]]

    def _parse_embedding_row(self, row):
        """Parse a row in the general_embeddings file"""
        content = row.rstrip().split(" ")
        return content[0], [float(i) for i in content[1:]]

    def _process_word_embedding(
        self,
        word,
        embedding,
        vocab,
        word_to_id,
        general_word_embeddings,
    ):
        """Process a single word embedding"""
        if word in vocab and word not in word_to_id and not re.match(r"^\d+$", word):
            word_to_id[word] = len(general_word_embeddings)
            general_word_embeddings.append(embedding)

    def _handle_out_of_vocab_word(self, words):
        """Handle out-of-vocabulary words by random selection"""
        sampling_prob = (
            1
            / len(words)
            * np.ones(
                len(words),
            )
        )
        sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
        return words[sampling_index[0]]
