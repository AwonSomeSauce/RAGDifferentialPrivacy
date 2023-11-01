import pandas as pd
from mechanisms.detectors.santext_detector import SanTextDetector
from mechanisms.detectors.custext_detector import CusTextDetector
from mechanisms.detectors.presidio_detector import PresidioDetector
from mechanisms.custext import CusText
from mechanisms.santext import SanText


def main():
    data = {
        "sentence": [
            "I loved the movie because of its plot and character development. My number is 2506866708",
            "The acting was subpar, and I was really disappointed. I live in Dhaka, Bangladesh",
            "One of the best movies I've ever watched. Highly recommended!",
            "The storyline was predictable and lacked depth.",
            "Stunning visuals and outstanding performances by the lead actors.",
            "I wouldn't watch it again. The pace was too slow.",
            "A cinematic masterpiece by James Bond that's both touching and captivating.",
            "The soundtrack perfectly complemented the movie's tone.",
            "While the movie had a strong start, it failed to maintain that momentum.",
            "A decent one-time watch, but not something to rave about.",
        ]
    }
    df = pd.DataFrame(data)
    word_embedding = 'glove'
    word_embedding_path = "glove.42B.300d.txt"
    epsilon = 1.0
    top_k = 5

    santext_detector = SanTextDetector(0.9)
    custext_detector = CusTextDetector()
    presidio_detector = PresidioDetector()

    # Initialize and use CusText to sanitize dataset
    cus_text_mechanism = CusText(word_embedding, word_embedding_path, epsilon, top_k, custext_detector)
    cus_sanitized_df = cus_text_mechanism.sanitize(df)

    # Initialize and use CusText + Presidio to sanitize dataset
    cus_pre_text_mechanism = CusText(word_embedding, word_embedding_path, epsilon, top_k, presidio_detector)
    cus_pre_sanitized_df = cus_pre_text_mechanism.sanitize(df)

    # Initialize and use SanText to sanitize dataset
    san_text_mechanism = SanText(word_embedding, word_embedding_path, epsilon, 0.3, santext_detector)
    san_sanitized_df = san_text_mechanism.sanitize(df)

    # Initialize and use SanText + Presidio to sanitize dataset
    san_pre_text_mechanism = SanText(word_embedding, word_embedding_path, epsilon, 0.3, presidio_detector)
    san_pre_sanitized_df = san_pre_text_mechanism.sanitize(df)

    # Display the original and sanitized reviews side-by-side
    for original, cus_sanitized, cus_pre_sanitized, san_sanitized, san_pre_sanitized in zip(df.sentence, cus_sanitized_df.sentence, cus_pre_sanitized_df.sentence, san_sanitized_df.sentence, san_pre_sanitized_df.sentence):
        print("Original:", original)
        print("CusText:", cus_sanitized)
        print("CusText + Presidio:", cus_pre_sanitized)
        print("SanText:", san_sanitized)
        print("SanText + Presidio:", san_pre_sanitized)
        print("-" * 80)


if __name__ == "__main__":
    main()
