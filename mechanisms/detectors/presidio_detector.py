from collections import Counter
from presidio_analyzer import AnalyzerEngine
from typing import List, Union
from presidio_analyzer.nlp_engine import NlpEngineProvider

LANGUAGES_CONFIG_FILE = "mechanisms/detectors/conf/default.yaml"

class PresidioDetector:

    def __init__(self):
        provider = NlpEngineProvider(conf_file=LANGUAGES_CONFIG_FILE)
        self.engine = AnalyzerEngine(nlp_engine=provider.create_engine())

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
