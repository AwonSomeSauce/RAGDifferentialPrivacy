import pandas as pd
from mechanisms.custext import CusText
from mechanisms.santext import SanText


def main():
    # Sample dataset of 10 random movie reviews
    data = {
        "sentence": [
            "I loved the movie because of its plot and character development.",
            "The acting was subpar, and I was really disappointed.",
            "One of the best movies I've ever watched. Highly recommended!",
            "The storyline was predictable and lacked depth.",
            "Stunning visuals and outstanding performances by the lead actors.",
            "I wouldn't watch it again. The pace was too slow.",
            "A cinematic masterpiece that's both touching and captivating.",
            "The soundtrack perfectly complemented the movie's tone.",
            "While the movie had a strong start, it failed to maintain that momentum.",
            "A decent one-time watch, but not something to rave about.",
        ]
    }
    df = pd.DataFrame(data)

    # Parameters for CusText
    word_embedding = "glove"  # placeholder, replace with actual value
    word_embedding_path = "glove.42B.300d.txt"  # placeholder, replace with actual path
    epsilon = 3.0  # Adjust based on your desired privacy level
    top_k = 5  # Sample value, adjust as needed

    # Initialize and use CusText to sanitize dataset
    cus_text_mechanism = CusText(word_embedding, word_embedding_path, epsilon, top_k)
    cus_sanitized_df = cus_text_mechanism.sanitize(df)

    # Initialize and use SanText to sanitize dataset
    san_text_mechanism = SanText(word_embedding, word_embedding_path, epsilon, top_k, 0.2)
    san_sanitized_df = san_text_mechanism.sanitize(df)

    # Display the original and sanitized reviews side-by-side
    for original, cus_sanitized, san_sanitized in zip(df.sentence, cus_sanitized_df.sentence, san_sanitized_df.sentence):
        print("Original  :", original)
        print("CusText:", cus_sanitized)
        print("SanText:", san_sanitized)
        print("-" * 80)


if __name__ == "__main__":
    main()
