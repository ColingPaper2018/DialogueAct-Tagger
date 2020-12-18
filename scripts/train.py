import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import SVMConfig
from trainers.svm_trainer import SVMTrainer
from corpora.taxonomy import Taxonomy
import argparse
import datetime
import time
from corpora.Maptask.Maptask import Maptask
from corpora.AMI.AMI import AMI
from corpora.Switchboard.Switchboard import Switchboard
from pathlib import Path

if __name__ == "__main__":
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description='DialogueActTrain - Train a DA Tagger using '
                                                 'annotated dialogue corpora')
    parser.add_argument('-out-folder', dest='out', type=str, default=f"models/{time.time()}/",
                        help='where the model files will be stored')
    parser.add_argument('-taxonomy', dest='taxonomy', type=str, default="isotag",
                        help='which taxonomy to train. Options are: \n'
                             'isotag -- ISO standard (default)'
                             'amitag -- AMI corpus annotation'
                             'swdatag -- Switchboard DAMSL taxonomy')
    args = parser.parse_args()
    if args.out is None or args.taxonomy is None:
        parser.print_help(sys.stderr)
        exit(1)
    taxonomy = Taxonomy.from_str(args.taxonomy)
    cfg = SVMConfig(taxonomy=taxonomy, dep=True, indexed_dep=True, indexed_pos=True, prev=True, ngrams=True,
                    corpora_list=[(Maptask, str(Path("data/Maptask").resolve())),
                                  (AMI, str(Path("data/AMI/corpus").resolve())),
                                  (Switchboard, str(Path("data/Switchboard").resolve()))])
    cfg.out_folder = args.out
    d = SVMTrainer(cfg)
    tagger = d.train(args.out)
