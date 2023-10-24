class BaseMechanism:
    def __init__(self, word_embedding, word_embedding_path, epsilon):
        self.word_embedding = word_embedding
        self.word_embedding_path = word_embedding_path
        self.epsilon = epsilon

    def sanitize(self, dataset):
        raise NotImplementedError
 