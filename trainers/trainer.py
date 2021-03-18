from corpora.corpus import Corpus
from taggers.dialogue_act_tagger import DialogueActTagger
from config import Config
import logging
from corpora.taxonomy import Taxonomy
from typing import List, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class Trainer:
    def __init__(self, config: Config, taxonomy: Taxonomy, corpora_list: List[Tuple[type, str]]):
        self.config = config
        self.taxonomy = taxonomy
        self.corpora = []
        if len(corpora_list) == 0:
            logger.error("There are no corpora loaded, and the classifier won't train. "
                         "Please check README.md for information on how to obtain more data")
            exit(1)
        for corpus in corpora_list:
            try:
                assert (issubclass(corpus[0], Corpus))
            except AssertionError:
                logger.error("DialogueActTrain error - The corpora list contains objects which are not corpora")
                logger.error("Please ensure each of your corpora is a subclass of Corpus")
                exit(1)

    def train(self, out_file: str) -> DialogueActTagger:
        raise NotImplementedError()

