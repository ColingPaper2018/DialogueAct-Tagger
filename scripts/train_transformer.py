import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from corpora.taxonomy import Taxonomy
from corpora.maptask import Maptask
from corpora.switchboard import Switchboard
from corpora.ami import AMI
from corpora.midas import MIDAS
from corpora.daily_dialog import DailyDialog
from taggers.transformer_tagger import TransformerTagger
from trainers.transformer_trainer import TransformerTrainer
from config import TransformerConfig
import torch.optim as optim
from corpora.corpus import Utterance


if __name__ == "__main__":
    config = TransformerConfig(
        taxonomy=Taxonomy.ISO,
        device="cpu",
        optimizer=optim.Adam,
        lr=2e-5,
        n_epochs=1,
        batch_size=256,
        max_seq_len=128,
        out_folder="models/transformer_example/",
    )
    trainer = TransformerTrainer(
        config,
        corpora_list=[
            # (Maptask, str(Path("data/Maptask").resolve())),
            # (AMI, str(Path("data/AMI/corpus").resolve())),
            (Switchboard, str(Path("data/Switchboard").resolve())),
            (DailyDialog, str(Path("data/DailyDialog").resolve())),
        ],
    )
    t = trainer.train()
