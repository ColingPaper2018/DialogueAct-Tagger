from enum import Enum
from trainers.svm_trainer import SVMTrain
from corpora.Oasis.Oasis import Oasis
from corpora.Switchboard.Switchboard import Switchboard
from corpora.AMI.AMI import AMI
from corpora.VerbMobil.VerbMobil import VerbMobil
from corpora.Maptask.Maptask import Maptask
from pathlib import Path
import logging
import os, datetime, time, json


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

class BertConfig:
    def __init__(self):
        self.fine_tuning = "full"  # "full" means backprop into transformer
        self.output_type = "token"  # sentence- or token-level 
        self.batch_size = 32
        self.device = "cpu"
        self.lower_case = True
        self.weight_decay_rate = 0.01
        self.learning_rate = 3e-5
        self.n_epochs = 4
        self.clip_grad_norm = 1.0
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        self.save_path = os.path.join(os.path.abspath("."), "models/bert", timestamp)

    def to_json(self):
        with open(f"{self.save_path}/config.json", "w") as config_f:
            json.dump({
                # Model type
                "model_type": self.model_type.name,
                "weight_decay_rate": self.weight_decay_rate,
                "learning_rate": self.learning_rate,
                "clip_grad_norm": self.clip_grad_norm,
                "n_epochs": self.n_epochs,
                "fine_tuning": self.fine_tuning,
                "lower_case": self.lower_case,
                "batch_size": self.batch_size
            }, config_f, indent=4)

    @staticmethod
    def from_json(json_file):
        with open(json_file) as config_f:
            config_dict = json.load(config_f)
        conf = BertConfig()

        # Model Type
        conf.model_type = config_dict.get("model_type")
        conf.weight_decay_rate = config_dict.get("weight_decay_rate")
        conf.learning_rate = config_dict.get("learning_rate")
        conf.clip_grad_norm = config_dict.get("clip_grad_norm")
        conf.n_epochs = config_dict.get("n_epochs")
        conf.fine_tuning = config_dict.get("fine_tuning")
        conf.lower_case = config_dict.get("lower_case")
        conf.batch_size = config_dict.get("batch_size")

