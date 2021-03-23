from config import CONFIG
import logging
from logzero import setup_logger
import sys

from gensim.test.utils import common_texts
from gensim.models import Word2Vec




if __name__ == "__main__":
    debug = True
    logger = setup_logger(__file__, level=logging.INFO)
    if debug:
        logger.info(f"{common_texts}")
        sys.exit(0)
    our_model = Word2Vec(common_texts, size=10, window=5, min_count=1, workers=4)
    
    our_model.save(str(CONFIG.data / "intermediate" / "tempmodel.w2v"))

    logger.info(our_model.wv.most_similar("computer", topn=5))
    logger.info(our_model["computer"])

