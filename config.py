from enum import Enum
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class Model(Enum):
    SVM = "SVM"


class Config:
    def __init__(self, model_type: Model):
        self.model_type = model_type
        self.out_folder = "models/svm/"
        self.acceptance_threshold = 0.5

    @staticmethod
    def from_dict(dict_):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()


class SVMConfig(Config):
    def __init__(self, indexed_pos: bool, indexed_dep: bool,
                 ngrams: bool, dep: bool, prev: bool):
        Config.__init__(self, Model.SVM)
        self.classifier = CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)
        self.indexed_pos = indexed_pos
        self.indexed_dep = indexed_dep
        self.ngrams = ngrams
        self.dep = dep
        self.prev = prev

    @staticmethod
    def from_dict(dict_):
        return SVMConfig(
            indexed_pos=dict_['indexed_pos'],
            dep=dict_['dep'],
            prev=dict_['prev'],
            indexed_dep=dict_['indexed_dep'],
            ngrams=dict_['ngrams'],
        )

    def to_dict(self):
        return {
            "indexed_pos": self.indexed_pos,
            "dep": self.dep,
            "ngrams": self.ngrams,
            "prev": self.prev,
            "indexed_dep": self.indexed_dep
        }
