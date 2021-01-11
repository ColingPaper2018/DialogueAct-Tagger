from enum import Enum
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import logging
from corpora.taxonomy import Taxonomy
from typing import List, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class Model(Enum):
    SVM = "SVM"


class Config:
    def __init__(self, model_type: Model, taxonomy: Taxonomy, corpora_list: List[Tuple[type, str]]):
        current_timestamp = time.time()
        self.model_type = model_type
        self.out_folder = f"models/{current_timestamp}/"
        self.acceptance_threshold = 0.5
        self.taxonomy = taxonomy
        self.corpora_list = corpora_list

    @staticmethod
    def from_dict(dict_):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()


class SVMConfig(Config):
    def __init__(self, taxonomy: Taxonomy, indexed_pos: bool, indexed_dep: bool,
                 ngrams: bool, dep: bool, prev: bool, corpora_list: List[Tuple[type, str]],
                 pipeline_files: List[str] = None):
        Config.__init__(self, Model.SVM, taxonomy, corpora_list)
        if pipeline_files is None:
            pipeline_files = []
        self.classifier = CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)
        self.indexed_pos = indexed_pos
        self.indexed_dep = indexed_dep
        self.ngrams = ngrams
        self.dep = dep
        self.prev = prev
        self.taxonomy = taxonomy
        self.pipeline_files = pipeline_files

    @staticmethod
    def from_dict(dict_):
        svm_config = SVMConfig(
            indexed_pos=dict_['indexed_pos'],
            dep=dict_['dep'],
            prev=dict_['prev'],
            indexed_dep=dict_['indexed_dep'],
            ngrams=dict_['ngrams'],
            taxonomy=Taxonomy.from_str(dict_['taxonomy']),
            pipeline_files=dict_['pipeline_files']
        )
        svm_config.out_folder = dict_['out_folder']
        return svm_config

    def to_dict(self):
        return {
            "indexed_pos": self.indexed_pos,
            "dep": self.dep,
            "prev": self.prev,
            "indexed_dep": self.indexed_dep,
            "ngrams": self.ngrams,
            "taxonomy": self.taxonomy.to_str(),
            "pipeline_files": self.pipeline_files,
            "out_folder": self.out_folder
        }
