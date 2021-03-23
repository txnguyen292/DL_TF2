from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from logzero import setup_logger

documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]

if __name__ == "__main__":
    logger = setup_logger(__file__, level=logging.INFO)
    tfidf = TfidfVectorizer()
    bow_rep_tfidf = tfidf.fit_transform(processed_docs)
    logger.info(f"IDF for all words in the vocabulary {tfidf.idf_}")
    logger.info(f"All words in the vocabulary {tfidf.get_feature_names()}")
    logger.info(f"TFIDF representation for all documents in our corpus {bow_rep_tfidf.toarray()}")
    

