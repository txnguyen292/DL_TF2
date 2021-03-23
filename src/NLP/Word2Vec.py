from gensim.models import Word2Vec, KeyedVectors

from config import CONFIG

pretrainedPath = CONFIG.data / "external" / "GoogleNews-vectors-negative300.bin"

if __name__ == "__main__":
    w2v_model = KeyedVectors.load_word2vec_format(str(pretrainedPath), binary=True)
    print("Done loading Word2Vec")
    print(len(w2v_model.vocab)) #Number of words in the vocabulary
    print(w2v_model.most_similar["beautiful"])
    print(w2v_model["beautiful"])