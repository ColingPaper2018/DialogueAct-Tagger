import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from config import Config
import argparse
import time
import datetime

if __name__ == "__main__":
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    cfg = Config()
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
    d = cfg.get_trainer_inst()
    d.train(args.layer, args.out)
