from enum import Enum
from trainers.svm_trainer import SVMTrain
from corpora.Oasis.Oasis import Oasis
from corpora.Switchboard.Switchboard import Switchboard
from corpora.AMI.AMI import AMI
from corpora.VerbMobil.VerbMobil import VerbMobil
from corpora.Maptask.Maptask import Maptask
from pathlib import Path
import logging


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
