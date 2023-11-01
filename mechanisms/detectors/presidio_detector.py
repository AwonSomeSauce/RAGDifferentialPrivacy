from collections import Counter
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from typing import List, Union

class PresidioDetector:

    def __init__(self):
        self.engine = AnalyzerEngine(nlp_engine=NlpEngineProvider().create_engine())

    def detect(self, vocab: Union[List[str], Counter]) -> List[str]:
        # Convert vocab to list of words if it's a Counter object
        if isinstance(vocab, Counter):
            vocab = list(vocab.keys())

        sensitive_words = set()

        # Analyze each word in the vocab
        for word in vocab:
            results = self.engine.analyze(text=str(word), language='en')
            if results:
                sensitive_words.add(word)

        return list(sensitive_words)
