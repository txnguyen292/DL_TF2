from bs4 import BeautifulSoup
from urllib.request import urlopen
import logging
from logzero import setup_logger

from config import CONFIG

loglvl = dict(info=logging.INFO, debug=logging.INFO, warning=logging.WARNING)

myurl = """https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python"""




if __name__ == "__main__":
    logger = setup_logger(__file__, level=logging.DEBUG)
    html = urlopen(myurl).read()
    logger.debug(html)
    soupified = BeautifulSoup(html, "html.parser")
    question = soupified.find("div", {"class": "question"})
    questionText = question.find("div", {"class": "s-prose js-post-body"})
    logger.debug(question)
    logger.info("Question: \n", questionText.get_text().strip())
    answer = soupified.find("div", {"class": "answer"})
    answerText = answer.find("div", {"class": "s-prose js-post-body"})
    logger.info("Best answer: \n", answerText.get_text().strip())