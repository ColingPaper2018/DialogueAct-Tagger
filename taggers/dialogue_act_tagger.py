from corpora.corpus import Utterance, Tag
from config import Config
from typing import List


class DialogueActTagger:
    def __init__(self, config: Config):
        self.config = config

    def tag(self, utterance: Utterance) -> List[Tag]:
        raise NotImplementedError()
