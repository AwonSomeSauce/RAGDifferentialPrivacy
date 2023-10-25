import unicodedata
from collections import Counter
from tqdm import tqdm
import numpy as np
import random
from spacy.lang.en import English
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
from .base_mechanism import BaseMechanism

class SanText(BaseMechanism):
    def __init__(self, word_embedding, word_embedding_path, epsilon, sensitive_word_percentage, p):
        super().__init__(word_embedding, word_embedding_path, epsilon)
        self.sensitive_word_percentage = sensitive_word_percentage
        self.p = p

    @staticmethod
    def normalize_word(text):
        return unicodedata.normalize('NFD', text)

    def build_vocab_from_dataset(self, df, tokenizer):
        vocab = Counter()

        for text in df['sentence']:
            tokenized_text = [token.text for token in tokenizer(text)]
            for token in tokenized_text:
                vocab[token] += 1

        return vocab

    def compute_probability_matrix(self, word_embed_1, word_embed_2):
        distance = euclidean_distances(word_embed_1, word_embed_2)
        sim_matrix = -distance
        prob_matrix = softmax(self.epsilon * sim_matrix / 2, axis=1)
        return prob_matrix

    def process_word_embeddings(self, vocab):
        word_to_id, sensitive_word_to_id = {}, {}
        general_word_embeddings, sensitive_word_embeddings = [], []

        num_lines = sum(1 for _ in open(self.word_embedding_path))

        with open(self.word_embedding_path) as file:
            # Handle potential header in word embeddings file
            if len(file.readline().split()) != 2:
                file.seek(0)

            for row in tqdm(file, total=num_lines - 1):
                content = row.rstrip().split(' ')
                current_word = self.normalize_word(content[0])
                embedding = [float(i) for i in content[1:]]

                if current_word in vocab and current_word not in word_to_id:
                    word_to_id[current_word] = len(general_word_embeddings)
                    general_word_embeddings.append(embedding)
                    
                    if current_word in self.sensitive_words_to_id:
                        sensitive_word_to_id[current_word] = len(sensitive_word_embeddings)
                        sensitive_word_embeddings.append(embedding)

        return np.array(general_word_embeddings), np.array(sensitive_word_embeddings), word_to_id, sensitive_word_to_id

    def transform_sentences(self, df):
        tokenizer = English()
        vocab = self.build_vocab_from_dataset(df, tokenizer)
        
        # Identify sensitive words
        num_sensitive_words = int(self.sensitive_word_percentage * len(vocab))
        words = [key for key, _ in vocab.most_common()]
        sensitive_words = words[-num_sensitive_words - 1:]
        self.sensitive_words_to_id = {word: idx for idx, word in enumerate(sensitive_words)}

        general_embeddings, sensitive_embeddings, word_to_id, sensitive_word_to_id = self.process_word_embeddings(vocab)
        prob_matrix = self.compute_probability_matrix(general_embeddings, sensitive_embeddings)
        # Process sentences and apply transformations
        sanitized_sentences = [self.sanitize_sentence(row['sentence'], tokenizer, word_to_id, sensitive_word_to_id, prob_matrix, words) for _, row in df.iterrows()]

        sanitized_df = df.copy()
        sanitized_df['sentence'] = sanitized_sentences

        return sanitized_df

    def sanitize_sentence(self, sentence, tokenizer, word_to_id, sensitive_word_to_id, prob_matrix, all_words):
        tokens = [token.text for token in tokenizer(sentence)]
        sanitized_tokens = []
        id2sword = {v: k for k, v in sensitive_word_to_id.items()}

        for word in tokens:
            if word in word_to_id:
                if word in sensitive_word_to_id:
                    sanitized_tokens.append(self.get_substitute_word(word, word_to_id, prob_matrix, id2sword))
                else:
                    if random.random() <= self.p:
                        sanitized_tokens.append(self.get_substitute_word(word, word_to_id, prob_matrix, id2sword))
                    else:
                        sanitized_tokens.append(word)
            else:
                # Handle out-of-vocab words
                sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                sanitized_tokens.append(all_words[sampling_index[0]])

        return " ".join(sanitized_tokens)

    def get_substitute_word(self, word, word_to_id, prob_matrix, id2sword):
        word_idx = word_to_id[word]
        sampling_prob = prob_matrix[word_idx]
        substitute_idx = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
        return id2sword[substitute_idx[0]]

    def sanitize(self, dataset):
        return self.transform_sentences(dataset)
