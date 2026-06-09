import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        self.nlp = spacy.load(model_name)

    def get_embedding(self, text):
        return self.nlp(text).vector.tolist()

    def get_similarity(self, word1, word2):
        return self.nlp(word1).similarity(self.nlp(word2))