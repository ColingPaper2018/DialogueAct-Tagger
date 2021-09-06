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
    def __init__(self, model_type: Model, taxonomy: Taxonomy, out_folder: str = None):
        current_timestamp = time.time()
        self.model_type = model_type
        if out_folder is None:
            out_folder = f"models/{current_timestamp}/"
        self.out_folder = out_folder
        self.acceptance_threshold = 0.5
        self.taxonomy = taxonomy

    @staticmethod
    def from_dict(dict_):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()


class SVMConfig(Config):
    def __init__(
        self,
        taxonomy: Taxonomy,
        indexed_pos: bool,
        indexed_dep: bool,
        ngrams: bool,
        dep: bool,
        pos: bool,
        prev: bool,
        pipeline_files: List[str] = None,
        out_folder: str = None,
    ):
        Config.__init__(self, Model.SVM, taxonomy, out_folder)
        if pipeline_files is None:
            pipeline_files = []
        self.indexed_pos = indexed_pos
        self.indexed_dep = indexed_dep
        self.ngrams = ngrams
        self.dep = dep
        self.pos = pos
        self.prev = prev
        self.taxonomy = taxonomy
        self.pipeline_files = pipeline_files

    @staticmethod
    def create_classifier():
        return CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)

    @staticmethod
    def from_dict(dict_):
        svm_config = SVMConfig(
            indexed_pos=dict_["indexed_pos"],
            dep=dict_["dep"],
            pos=dict_["pos"],
            prev=dict_["prev"],
            indexed_dep=dict_["indexed_dep"],
            ngrams=dict_["ngrams"],
            taxonomy=Taxonomy.from_str(dict_["taxonomy"]),
            pipeline_files=dict_["pipeline_files"],
        )
        svm_config.out_folder = dict_["out_folder"]
        return svm_config

    def to_dict(self):
        return {
            "indexed_pos": self.indexed_pos,
            "dep": self.dep,
            "pos": self.pos,
            "prev": self.prev,
            "indexed_dep": self.indexed_dep,
            "ngrams": self.ngrams,
            "taxonomy": self.taxonomy.to_str(),
            "pipeline_files": self.pipeline_files,
            "out_folder": self.out_folder,
        }


class TransformerConfig(Config):
    def __init__(
        self,
        taxonomy: Taxonomy,
        device: str,
        optimizer: type,
        lr: float,
        batch_size: int,
        max_seq_len: int,
        n_epochs: int,
        pipeline_files: Optional[List[str]] = None,
        out_folder: str = None,
    ):
        Config.__init__(self, Model.SVM, taxonomy, out_folder)
        if pipeline_files is None:
            pipeline_files = []
        self.device = device
        self.lr = lr
        self.optimizer = optimizer
        self.taxonomy = taxonomy
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.unk_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        self.n_epochs = n_epochs
        self.pipeline_files = pipeline_files

    @staticmethod
    def from_dict(dict_):
        transformer_config = TransformerConfig(
            taxonomy=Taxonomy.from_str(dict_["taxonomy"]),
            device=dict_["device"],
            optimizer=dict_["optimizer"],
            lr=dict_["lr"],
            batch_size=dict_["batch_size"],
            max_seq_len=dict_["max_seq_len"],
            n_epochs=dict_["n_epochs"],
            pipeline_files=dict_["model_files"],
        )
        transformer_config.out_folder = dict_["out_folder"]
        return transformer_config

    def to_dict(self):
        return {
            "device": self.device,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "taxonomy": self.taxonomy.to_str(),
            "out_folder": self.out_folder,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "pad_index": self.pad_index,
            "unk_index": self.unk_index,
            "n_epochs": self.n_epochs,
            "pipeline_files": self.pipeline_files,
        }
