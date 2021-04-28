from taggers.dialogue_act_tagger import DialogueActTagger

from typing import List
from corpora.corpus import Corpus
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import random


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
        random.shuffle(self.test_set)
        self.test_set = self.test_set[0:100]

    def test(self, tagger: DialogueActTagger):
        y_true = [u.tags for u in self.test_set]
        y_pred = tagger.tag_batch(self.test_set)

        if "dimension" in tagger.classifiers:
            # 1) Compare dimension results
            y_dim_true = [[t.dimension.value for t in tags] for tags in y_true]
            y_dim_pred = [[t.dimension.value for t in tags] for tags in y_pred]
            binarizer = MultiLabelBinarizer()
            binarizer.fit(y_dim_true + y_dim_pred)

            # target_names = list(tagger.config.taxonomy.value.get_dimension_taxonomy().values().values())
            # labels = list(tagger.config.taxonomy.value.get_dimension_taxonomy().values().keys())
            # print("TARGET:", target_names)
            # print("LABELS:", labels)
            #

            print("Dimension Classification Report")
            print(classification_report(binarizer.transform(y_dim_true), binarizer.transform(y_dim_pred)))
            for dimension in tagger.config.taxonomy.value.get_dimension_taxonomy():
                if dimension.value > 0:
                    y_comm_true = []
                    y_comm_pred = []
                    for idx, datapoint in enumerate(y_true):
                        if any(t.dimension == dimension for t in datapoint):
                            y_comm_true.append([t.comm_function.value for t in datapoint if t.dimension == dimension][0])
                            try:
                                y_comm_pred.append([t.comm_function.value for t in y_pred[idx] if t.dimension == dimension][0])
                            except IndexError:
                                y_comm_pred.append(0)  # unknown
                    print(f"Communication Function Report for {dimension}")
                    print(classification_report(y_comm_true, y_comm_pred))
                    # labels=labels, target_names=target_names))

    def test_compare(self, taggers: List[DialogueActTagger]):
        raise NotImplementedError()
