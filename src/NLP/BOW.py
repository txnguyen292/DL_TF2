from typing import Counter
from sklearn.feature_extraction.text import CountVectorizer
import logging
from logzero import setup_logger

documents = ["Dog bites man.", "Man bites dog.",
               "Dog eats meat.", "Man eats food."]

processed_docs = [doc.lower().replace(".", "") for doc in documents]




if __name__ == "__main__":
    logger = setup_logger(__file__, level=logging.INFO)
    count_vect = CountVectorizer()
    bow_rep = count_vect.fit_transform(processed_docs)
    print("Our Vocabulary:", count_vect.vocabulary_)

    logger.info(f"Bow representation for 'dog bites man':{bow_rep[0].toarray()}")
    logger.info(f"Bow representation for 'man bites dog': {bow_rep[1].toarray()}")

    temp = count_vect.transform(["dog and dog are friends"])
    logger.info(f"Bow representation for 'dog and dog are friends': {temp.toarray()}")

