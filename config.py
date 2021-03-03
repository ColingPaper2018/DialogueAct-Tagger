from enum import Enum
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import logging
from corpora.taxonomy import Taxonomy
from typing import List, Tuple, Optional
import time
from transformers import BertTokenizer

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
            pipeline_files=dict_['pipeline_files'],
            corpora_list=dict_['corpora_list']
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
            "out_folder": self.out_folder,
            "corpora_list": self.corpora_list
        }


class TransformerConfig(Config):
    def __init__(self, taxonomy: Taxonomy, corpora_list: List[Tuple[type, str]], device: str,
                 optimizer: type, lr: float, batch_size: int, max_seq_len: int, n_epochs: int,
                 pipeline_files: Optional[List[str]] = None):
        Config.__init__(self, Model.SVM, taxonomy, corpora_list)
        if pipeline_files is None:
            pipeline_files = []
        self.device = device
        self.lr = lr
        self.optimizer = optimizer
        self.taxonomy = taxonomy
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.unk_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        self.n_epochs = n_epochs
        self.pipeline_files = pipeline_files

    @staticmethod
    def from_dict(dict_):
        transformer_config = TransformerConfig(
            taxonomy=Taxonomy.from_str(dict_['taxonomy']),
            corpora_list=dict_['corpora_list'],
            device=dict_['device'],
            optimizer=dict_['optimizer'],
            lr=dict_['lr'],
            batch_size=dict_['batch_size'],
            max_seq_len=dict_['max_seq_len'],
            n_epochs=dict_['n_epochs'],
            pipeline_files=dict_['model_files']
        )
        transformer_config.out_folder = dict_['out_folder']
        return transformer_config

    def to_dict(self):
        return {
            "device": self.device,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "taxonomy": self.taxonomy.to_str(),
            "out_folder": self.out_folder,
            "corpora_list": self.corpora_list,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "pad_index": self.pad_index,
            "unk_index": self.unk_index,
            "n_epochs": self.n_epochs,
            "pipeline_files": self.pipeline_files
        }
