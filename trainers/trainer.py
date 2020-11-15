from corpora.Corpus import Corpus
from typing import List
from config import Config
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class Trainer:
    def __init__(self, corpora_list: List[Corpus], config: Config):
        self.corpora = []
        self.config = config
        for corpus in corpora_list:
            try:
                assert (issubclass(type(corpus), Corpus))
            except AssertionError:
                logger.error("DialogueActTrain error - The corpora list contains objects which are not corpora")
                logger.error("Please ensure each of your corpora is a subclass of Corpus")
                exit(1)
            if corpus.utterances is not None and len(corpus.utterances) > 0:  # corpus loaded successfully
                if corpus.taxonomy == config.taxonomy:
                    self.corpora.append(corpus)
                else:
                    logger.warning(f"Corpus {corpus.name} not loaded. Corpus is using taxonomy {corpus.taxonomy}, "
                                   f"while this trainer is using taxonomy {config.taxonomy}")
        if len(self.corpora) == 0:
            logger.error("There are no corpora loaded, and the classifier won't train. "
                         "Please check README.md for information on how to obtain more data")
            exit(1)

    def train(self, out_file: str) -> Tagger:
        raise NotImplementedError()

