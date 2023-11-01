from collections import Counter

class SanTextDetector():
    """SanText sensitive word detection class"""
    def __init__(self, sensitive_word_percentage):
        self.sensitive_word_percentage = sensitive_word_percentage

    def detect(self, vocab: Counter):
        if not vocab:
            return []

        num_sensitive_words = int(self.sensitive_word_percentage * len(vocab))
        words = [word for word, _ in vocab.most_common()] 
        return words[-num_sensitive_words:]
