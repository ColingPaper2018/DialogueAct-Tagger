from enum import Enum
from trainers.svm_trainer import SVMTrain
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class Model(Enum):
    SVM = "SVM"


class Config:
    def __init__(self, corpora):
        self.model_type = Model.SVM
        self.out_folder = "models/svm/"
        self.acceptance_threshold = 0.5

        # NOTE adding corpora here will extend the dataset for training
        self.corpora = corpora
        logger.info("Corpora loaded succesfully! Loaded corpora:")
        logger.info([corpus.get_corpus_name() for corpus in self.corpora if corpus.csv_corpus is not None])
        
    def get_trainer_inst(self):
        if self.model_type == Model.SVM:
            return SVMTrain(self.corpora)
        else:
            raise NotImplementedError("The required model type is not supported yet")

    @staticmethod
    def from_json(json_file):
        with open(json_file) as f:
            c = Config([])
            config = json.load(f)
            c.out_folder = config['out_folder']
            c.corpora = config['corpora']
            c.acceptance_threshold = config['acceptance_threshold']
            if config['model_type'] == "SVM":
                c.model_type = Model.SVM
                return c
            else:
                raise NotImplementedError(f"Model {config.model_type} is not supported")

    def to_dict(self):
        c = {
            "corpora": [c.get_corpus_name() for c in self.corpora],
            "out_folder": self.out_folder,
            "acceptance_threshold": self.acceptance_threshold
        }
        if self.model_type == Model.SVM:
            c['model_type'] = "SVM"
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not serializable")
        return c

