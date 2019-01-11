from enum import Enum
from trainers.svm_trainer import SVMTrain
from corpora.Oasis.Oasis import Oasis
from corpora.Switchboard.Switchboard import Switchboard
from corpora.AMI.AMI import AMI
from corpora.VerbMobil.VerbMobil import VerbMobil
from corpora.Maptask.Maptask import Maptask
from pathlib import Path
import logging]
import os, datetime, time, json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class Model(Enum):
    SVM = 1


class Config:
    def __init__(self):
        self.model_type = Model.SVM
        self.out_folder = "models/svm/"

        # NOTE adding corpora here will extend the dataset for training
        self.corpora = [
                            AMI(str(Path("data/AMI/corpus").resolve())),
                            # Oasis(str(Path("data/Oasis").resolve())),
                            # Switchboard(str(Path("data/Switchboard").resolve())),
                            # VerbMobil((str(Path("data/Verbmobil").resolve()))),
                            Maptask(str(Path("data/Maptask/maptaskv2-1").resolve()))]
        logger.info("Corpora loaded succesfully! Loaded corpora:")
        logger.info([corpus.get_corpus_name() for corpus in self.corpora])
        
    def get_trainer_inst(self):
        if self.model_type == Model.SVM:
            return SVMTrain(self.corpora)
        else:
            raise NotImplementedError("The required model type is not supported yet")

class BertConfig:
    def __init__(self):
        self.fine_tuning = "full"  # "full" means backprop into transformer
        self.output_type = "token"  # sentence- or token-level prediction
        self.device = "cpu"
        self.lower_case = True
        self.weight_decay_rate = 0.01
        self.learning_rate = 3e-5
        self.n_epochs = 20
        self.clip_grad_norm = 1.0
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        self.save_path = os.path.join(os.path.abspath("."), "models/bert", self.timestamp)

    def to_json(self):
        with open(f"{self.save_path}/config.json", "w") as config_f:
            json.dump({
                # Model type
                "model_type": self.model_type.name,
                "weight_decay_rate" = self.weight_decay_rate,
                "learning_rate" = self.learning_rate,
                "clip_grad_norm" = self.clip_grad_norm,
                "n_epochs" = self.n_epochs,
                "fine_tuning" = self.fine_tuning,
                "lower_case" = self.lower_case
            })
