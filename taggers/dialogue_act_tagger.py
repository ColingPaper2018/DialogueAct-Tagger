from corpora.corpus import Utterance, Tag
from config import Config
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Classifier:
    model: Any
    target_tagset: dataclass


class DialogueActTagger:
    def __init__(self, config: Config):
        self.config = config
        self.classifiers: Dict[str, Classifier] = {}

    def tag(self, utterance: Utterance) -> List[Tag]:
        raise NotImplementedError()
