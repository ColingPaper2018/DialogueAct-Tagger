import os
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from taggers.svm_tagger import SVMTagger
from config import SVMConfig
from corpora.corpus import Utterance
from typing import List
import json
from utils import stringify_tags

from .trainer import Trainer
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class SVMTrainer(Trainer):
    def __init__(self, config: SVMConfig):
        Trainer.__init__(self, config, config.taxonomy)
        for c in config.corpora_list:
            try:
                self.corpora.append(c[0](c[1], config.taxonomy))
            except Exception as e:
                logger.warning(f"Corpus {c[0]} not loaded. {e}")

    @staticmethod
    def train_pipeline(config: SVMConfig, dataset: List[Utterance]):
        if all(len(u.tags) == 1 for u in dataset) and all(u.tags[0] == dataset[0].tags[0] for u in dataset):
            logger.warning(f"The only tag available for this classifier is {dataset[0].tags[0]}."
                           "The classifier will still be trained, but it won't recognise any other labels."
                           "Please provide additional data to obtain a working classifier. You can check README.md "
                           "for information on how to obtain more data")
            for _ in range(0, 3):
                dataset.append(Utterance(text="<<unk>>", context=[], tags=["<<unk>>"], speaker_id=0))
        features = SVMTagger.build_features(dataset, config)
        train_pipeline = Pipeline([
            # Use FeatureUnion to combine the features from wordcount and labels
            ('union', FeatureUnion(
                transformer_list=[('feature_' + str(i), pipeline) for i, pipeline in enumerate(features[1])]
            )),
            # Use a SVC classifier on the combined features
            ('classifier', config.classifier)
        ])
        if len(dataset) == 0:
            logger.error(f"Not enough data to train the classifier! Please check README.md for "
                         f"more information on how to obtain more data")
            return
        train_pipeline.fit(features[0], [u.tags for u in dataset])
        for _ in range(1, 4):
            del(dataset[-1])
        return train_pipeline

    def dump_model(self, pipelines: dict):
        # Create directory
        path = Path(os.path.dirname(self.config.out_folder))
        print(f"creating {self.config.out_folder}")
        path.mkdir(parents=True, exist_ok=True)

        # Save the config file
        with open(f"{self.config.out_folder}/config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)

        # Save the pipelines
        for pipeline in pipelines.keys():
            pickle.dump(pipelines[pipeline], open(f"{self.config.out_folder}/{pipeline}", 'wb'))
        return

    def train(self, dump=True):
        logger.info(f"Training Dialogue Act Tagger for {self.config.taxonomy} taxonomy, using the following corpora:"
                    f"{[c.name for c in self.corpora]}")
        dataset = []
        pipelines = {}

        for corpus in self.corpora:
                dataset = dataset + corpus.utterances
        if "dimension" in self.config.taxonomy.value.__annotations__.keys():
            # Train dimension tagger
            logger.info("Training dimension pipeline")
            dimension_dataset = stringify_tags(dataset, "dimension")
            pipelines['dimension'] = self.train_pipeline(self.config, dimension_dataset)

            # Train a comm-function classifier for each dimension
            dimension_labels = [
                [tag for tag in utt.tags]
                for utt in dimension_dataset
            ]
            dimension_values = list(set([label for tagset in dimension_labels for label in tagset]))
            for dimension_value in dimension_values:
                logger.info(f"Training communication function pipeline for dimension {dimension_value}")
                comm_dataset = stringify_tags(dataset, "comm_function",
                                              filter_attr="dimension", filter_value=dimension_value)
                pipelines[f'comm_{dimension_value}'] = self.train_pipeline(self.config, comm_dataset)
        else:
            logger.info("Training unified communication function pipeline")
            comm_dataset = stringify_tags(dataset, "comm_function")
            pipelines['comm_all'] = self.train_pipeline(self.config, comm_dataset)
        self.config.pipeline_files = list(pipelines.keys())
        if dump:
            self.dump_model(pipelines)
        return SVMTagger(self.config)
