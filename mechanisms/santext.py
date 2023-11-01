from collections import Counter
import random
import numpy as np
from tqdm import tqdm
from spacy.lang.en import English
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from .base_mechanism import BaseMechanism

class SanText(BaseMechanism):
    """Class for sanitizing text by using SanTextPlus mechanism"""
    def __init__(self, word_embedding, word_embedding_path, epsilon, p, detector):
        super().__init__(word_embedding, word_embedding_path, epsilon)
        self.p = p
        self.detector = detector

    def build_vocab_from_dataset(self, df, tokenizer):
        """Build vocabulary from dataset"""
        vocab = Counter()
        for text in df['sentence']:
            tokenized_text = [token.text for token in tokenizer(text)]
            vocab.update(tokenized_text)
        return vocab

    def compute_probability_matrix(self, word_embed_1, word_embed_2):
        """Compute the probability matrix based on euclidean distances between word embeddings"""
        distance = euclidean_distances(word_embed_1, word_embed_2)
        sim_matrix = -distance
        prob_matrix = softmax(self.epsilon * sim_matrix / 2, axis=1)
        return prob_matrix

    def process_word_embeddings(self, vocab):
        """Process word embeddings and return arrays and dictionaries for general and sensitive words"""
        word_to_id, sensitive_word_to_id = {}, {}
        general_word_embeddings, sensitive_word_embeddings = [], []
        num_lines = sum(1 for _ in open(self.word_embedding_path,  encoding="utf-8"))

        with open(self.word_embedding_path,  encoding="utf-8") as file:
            if not self._has_header(file):
                file.seek(0)
            num_lines = sum(1 for _ in file)
            file.seek(0)

            for row in tqdm(file, total=num_lines - 1):
                word, embedding = self._parse_embedding_row(row)
                self._process_word_embedding(word, embedding, vocab, word_to_id, general_word_embeddings, sensitive_word_to_id, sensitive_word_embeddings)

        return np.array(general_word_embeddings), np.array(sensitive_word_embeddings), word_to_id, sensitive_word_to_id

    def sanitize(self, dataset):
        """Sanitize dataset"""
        return self._transform_sentences(dataset)

    def _transform_sentences(self, df):
        """Transform sentences in the dataframe"""
        # Lowercase the entire 'sentence' column if using GloVe embeddings
        if self.word_embedding == 'glove':
            df['sentence'] = df['sentence'].str.lower()

        tokenizer = English()
        vocab = self.build_vocab_from_dataset(df, tokenizer)
        words = [key for key, _ in vocab.most_common()]
        sensitive_words = self.detector.detect(vocab)
        self.sensitive_words_to_id = {word: idx for idx, word in enumerate(sensitive_words)}
        processed_data = self.process_word_embeddings(vocab)
        general_embeddings, sensitive_embeddings, word_to_id, sensitive_word_to_id = processed_data
        prob_matrix = self.compute_probability_matrix(general_embeddings, sensitive_embeddings)
        sanitized_sentences = [self._sanitize_sentence(sentence, tokenizer, word_to_id, sensitive_word_to_id, prob_matrix, words) for sentence in df['sentence']]
        sanitized_df = df.copy()
        sanitized_df['sentence'] = sanitized_sentences
        return sanitized_df

    def _sanitize_sentence(self, sentence, tokenizer, word_to_id, sensitive_word_to_id, prob_matrix, all_words):
        """Sanitize individual sentence"""
        tokens = [token.text for token in tokenizer(sentence)]
        sanitized_tokens = []
        id_to_word = {v: k for k, v in sensitive_word_to_id.items()}
        for word in tokens:
            sanitized_tokens.append(self._get_word_substitute_or_original(word, word_to_id, sensitive_word_to_id, prob_matrix, id_to_word, all_words))
        return " ".join(sanitized_tokens)

    def _get_word_substitute_or_original(self, word, word_to_id, sensitive_word_to_id, prob_matrix, id_to_word, all_words):
        """Get substitute for word or return original if not sanitized"""
        if word in word_to_id:
            if word in sensitive_word_to_id or random.random() <= self.p:
                return self._get_substitute_word(word, word_to_id, prob_matrix, id_to_word)
            else:
                return word
        return self._handle_out_of_vocab_word(all_words)

    def _get_substitute_word(self, word, word_to_id, prob_matrix, id_to_word):
        """Retrieve a substitute word"""
        word_idx = word_to_id[word]
        sampling_prob = prob_matrix[word_idx]
        substitute_idx = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
        return id_to_word[substitute_idx[0]]

    def _parse_embedding_row(self, row):
        """Parse a row in the embeddings file"""
        content = row.rstrip().split(' ')
        return content[0], [float(i) for i in content[1:]]

    def _process_word_embedding(self, word, embedding, vocab, word_to_id, general_word_embeddings, sensitive_word_to_id, sensitive_word_embeddings):
        """Process a single word embedding"""
        word = word.lower()
        if word in vocab and word not in word_to_id:
            word_to_id[word] = len(general_word_embeddings)
            general_word_embeddings.append(embedding)
            if word in self.sensitive_words_to_id:
                sensitive_word_to_id[word] = len(sensitive_word_embeddings)
                sensitive_word_embeddings.append(embedding)

    def _handle_out_of_vocab_word(self, all_words):
        """Handle out-of-vocabulary words by random selection"""
        sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )
        sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
        return all_words[sampling_index[0]]
