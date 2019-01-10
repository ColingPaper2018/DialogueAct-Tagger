import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from config import Config
import argparse
import time
import json
import datetime
from corpora.Oasis.Oasis import Oasis
from corpora.Switchboard.Switchboard import Switchboard
from corpora.AMI.AMI import AMI
from corpora.VerbMobil.VerbMobil import VerbMobil
from corpora.Maptask.Maptask import Maptask
from pathlib import Path

if __name__ == "__main__":
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    cfg = Config([
        AMI(str(Path("data/AMI/corpus").resolve())),
        Oasis(str(Path("data/Oasis/corpus/Release").resolve())),
        Switchboard(str(Path("data/Switchboard/corpus").resolve())),
        VerbMobil(str(Path("data/Verbmobil").resolve()),
                  en_files=str(Path("corpora/VerbMobil/files.txt").resolve())),
        Maptask(str(Path("data/Maptask/maptaskv2-1").resolve()))])

    parser = argparse.ArgumentParser(description='DialogueActTrain - Train a DA Tagger using ISO-converted corpora')
    parser.add_argument('-out-folder', dest='out', type=str, default=f"models/{cfg.model_type}/",
                        help='where the model files will be stored')
    parser.add_argument('-class', dest='layer', type=str, default="all",
                        help='which level of the taxonomy to train. Options are: \n'
                             'all -- trains all the classifiers'
                             'dim -- trains the dimension classifier'
                             'task -- trains the task CF classifier'
                             'som -- trains the SOM CF classifier')
    args = parser.parse_args()
    if args.out is None:
        parser.print_help(sys.stderr)
        exit(1)
    cfg.out_folder = args.out
    d = cfg.get_trainer_inst()
    d.train(args.layer, args.out)
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=4)
