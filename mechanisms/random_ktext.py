import numpy as np
from scipy.spatial.distance import cdist


def get_replacement(w, D, sensitivity, epsilon):
    """
    Select a replacement for word w using the exponential mechanism, with utility based on Euclidean distance.

    :param w: The original word.
    :param D: A dictionary of word embeddings.
    :param sensitivity: The sensitivity of the exponential mechanism.
    :param epsilon: The privacy parameter.
    :return: A replacement word.
    """
    if w not in D:
        raise ValueError("Word not found in dictionary.")

    # Extract the embedding for the original word
    original_embedding = D[w].reshape(1, -1)  # Reshape for compatibility with cdist

    # Create an array of all other embeddings
    candidate_words = [word for word in D if word != w]
    candidate_embeddings = np.array([D[word] for word in candidate_words])

    # Compute Euclidean distances from the original word to all others
    distances = cdist(
        original_embedding, candidate_embeddings, metric="euclidean"
    ).flatten()

    # Convert distances to utilities. Since smaller distances indicate higher similarity (and thus utility),
    # we can invert the distances. Adjust this as needed for your specific utility definition.
    utilities = -distances

    # Apply the exponential mechanism to convert utilities to probabilities
    scaled_utilities = np.exp((epsilon / (2 * sensitivity)) * utilities)
    probabilities = scaled_utilities / np.sum(scaled_utilities)

    # Randomly select a replacement based on the computed probabilities
    replacement = np.random.choice(candidate_words, p=probabilities)
    return replacement


# Example usage
if __name__ == "__main__":
    # Load GloVe embeddings
    glove_embeddings = {}
    with open("glove.6B.300d.txt", "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            glove_embeddings[word] = vector

    # Get a replacement for a word
    # replacement_word = get_replacement("canada", glove_embeddings, k=5, sensitivity=1.0)
    # print(replacement_word)
    # Interactive part
    while True:
        input_word = input("Enter a word (type 'exit' to quit): ")
        if input_word.lower() == "exit":
            break
        print(
            get_replacement(input_word, glove_embeddings, sensitivity=3.14, epsilon=30)
        )
