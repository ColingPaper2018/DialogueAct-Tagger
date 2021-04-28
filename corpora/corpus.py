import os

from typing import List, Union, Dict
from corpora.taxonomy import Tag, Taxonomy


class Utterance:
    """
    Utterance provides an interface to interact with the individual
    sentences of a corpus
    """
    def __init__(self, text: str, tags: List[Union[Tag, str]],
                 context: List["Utterance"], speaker_id: int):
        self.text = text
        self.tags = tags
        self.context = context
        self.speaker_id = speaker_id

    def __str__(self):
        return f"{self.speaker_id}: {self.text} - {self.tags}"


class Corpus:
    def __init__(self, name: str, corpus_folder: str, taxonomy: Taxonomy):
        """
        A corpus is a collection of utterances encoded with a certain taxonomy. Every corpus has a source folder
        with its raw files, a method to validate these files, a method to load them, and a method to convert them
        into a target taxonomy using the Utterance interface
        :param corpus_folder: folder containing the corpus
        """
        if not os.path.isdir(corpus_folder):
            raise FileNotFoundError(f"The provided folder ({corpus_folder}) does not exist")
        self.name = name
        self.utterances: Dict[str, List[Utterance]] = {"train": [], "test": []}
        self.taxonomy = taxonomy
        return

    def validate_corpus(self, folder) -> bool:
        raise NotImplementedError()

    def load_corpus(self, folder):
        raise NotImplementedError()

    def parse_corpus(self, folder):
        raise NotImplementedError()

    def get_train_split(self):
        return self.utterances['train']

    def get_test_split(self):
        return self.utterances['test']

    @staticmethod
    def da_to_taxonomy(dialogue_act: str, taxonomy: Taxonomy, context) -> List[Tag]:
        raise NotImplementedError()

