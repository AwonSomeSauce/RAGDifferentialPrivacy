import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords


class CusTextDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def detect(self, vocab):
        return [word for word in vocab if word not in self.stop_words]
