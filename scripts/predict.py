import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from config import Config, Model
import argparse
from taggers.svm_tagger import SVMPredictor
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DialogueActTag - Tag a sentence with the ISO dialogue act taxonomy"
    )
    parser.add_argument(
        "-model", dest="model", type=str, help="the model folder to use for prediction"
    )
    parser.add_argument(
        "-class",
        dest="layer",
        type=str,
        default="all",
        help="which level of the taxonomy to tag. Options are: \n"
        "all -- trains all the classifiers (default)"
        "dim -- trains the dimension classifier"
        "task -- trains the task CF classifier"
        "som -- trains the SOM CF classifier",
    )
    parser.add_argument("-s", dest="sentence", type=str, help="the sentence to tag")
    parser.add_argument(
        "-p",
        dest="prev",
        type=str,
        help="[optional] the previous sentence in the dialogue",
    )

    args = parser.parse_args()
    if args.model is None or args.sentence is None:
        parser.print_help(sys.stderr)
        exit(1)

    logger.info("Restoring model config from meta.json")
    cfg = Config.from_json(f"{args.model}/meta.json")
    if cfg.model_type == Model.SVM:
        logger.info("Loading SVM tagger")
        tagger = SVMPredictor(cfg)
    else:
        raise NotImplementedError(f"Unknown classifier type: {cfg.model_type}")
    logger.info("Tagging utterance")
    print(tagger.dialogue_act_tag(args.sentence))
