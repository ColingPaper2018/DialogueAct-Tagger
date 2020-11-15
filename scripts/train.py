import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from config import SVMConfig
from trainers.svm_trainer import SVMTrainer
from corpora.taxonomy import Taxonomy
import argparse
import time
import json
import datetime
from corpora.AMI.AMI import AMI
from pathlib import Path

if __name__ == "__main__":
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    cfg = SVMConfig(dep=True, indexed_dep=True, indexed_pos=True, prev=True, ngrams=True)

    parser = argparse.ArgumentParser(description='DialogueActTrain - Train a DA Tagger using '
                                                 'annotated dialogue corpora')
    parser.add_argument('-out-folder', dest='out', type=str, default=f"models/{cfg.model_type}/",
                        help='where the model files will be stored')
    parser.add_argument('-taxonomy', dest='layer', type=str, default="all",
                        help='which taxonomy to train. Options are: \n'
                             'iso -- ISO standard'
                             'AMI -- AMI corpus annotation')
    args = parser.parse_args()
    if args.out is None:
        parser.print_help(sys.stderr)
        exit(1)
    cfg.out_folder = args.out
    d = SVMTrainer(corpora_list=[AMI(str(Path("data/AMI/corpus").resolve()),
                                     taxonomy=Taxonomy.from_str(args.taxonomy))], config=cfg)
    tagger = d.train(args.out)
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=4)
