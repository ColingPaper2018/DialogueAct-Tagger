from taggers.dialogue_act_tagger import DialogueActTagger

from typing import List
from corpora.corpus import Corpus
from sklearn.metrics import classification_report


class DialogueActTester:
    """
    A testing utility for Dialogue Act Tagger.
    Provides comparison of different DA tagging architectures on the same test set and
    in-depth statistics on a single classifier's performances
    """

    def __init__(self, corpora: List[Corpus]):
        self.test_set = []
        for corpus in corpora:
            self.test_set = self.test_set + corpus.get_test_split()

    def test(self, tagger: DialogueActTagger):
        y_true = [u.tags for u in self.test_set]
        y_pred = [tagger.tag(u) for u in self.test_set]
        return classification_report(y_true, y_pred)

    def test_compare(self, taggers: List[DialogueActTagger]):
        raise NotImplementedError()



