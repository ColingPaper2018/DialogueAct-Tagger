import os
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from taggers.svm_tagger import SVMTagger, DummyClassifier
from config import SVMConfig
from corpora.corpus import Utterance
from typing import List
import json
from utils import stringify_tags
from typing import Tuple

from .trainer import Trainer
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class SVMTrainer(Trainer):
    def __init__(self, config: SVMConfig, corpora_list: List[Tuple[type, str]]):
        Trainer.__init__(self, config, config.taxonomy, corpora_list)
        for c in corpora_list:
            self.corpora.append(c[0](c[1], config.taxonomy))

    @staticmethod
    def train_pipeline(config: SVMConfig, dataset: List[Utterance], n_classes: int = 1):
        logger.info(
            f"Tagset for this classifier: {set([t for u in dataset for t in u.tags])}"
        )

        if len(dataset) == 0:
            logger.warning(
                f"No data available for this classifier. A dummy classifier will be created."
                "Please provide additional data to obtain a working classifier. You can check README.md "
                "for information on how to obtain more data"
            )
            return DummyClassifier(n_classes=1)
        if all(len(u.tags) == 1 for u in dataset) and all(
            u.tags[0] == dataset[0].tags[0] for u in dataset
        ):
            logger.warning(
                f"The only tag available for this classifier is {dataset[0].tags[0]}."
                "The classifier will still be trained, but it won't recognise any other labels."
                "Please provide additional data to obtain a working classifier. You can check README.md "
                "for information on how to obtain more data"
            )
            return DummyClassifier(n_classes=1)
        features = SVMTagger.build_features(dataset, config)
        train_pipeline = Pipeline(
            [
                # Use FeatureUnion to combine the features from wordcount and labels
                (
                    "union",
                    FeatureUnion(
                        transformer_list=[
                            ("feature_" + str(i), pipeline)
                            for i, pipeline in enumerate(features[1])
                        ]
                    ),
                ),
                # Use a SVC classifier on the combined features
                ("classifier", config.create_classifier()),
            ]
        )
        if len(dataset) == 0:
            logger.error(
                f"Not enough data to train the classifier! Please check README.md for "
                f"more information on how to obtain more data"
            )
            return
        train_pipeline.fit(features[0], np.ravel([u.tags for u in dataset]))
        return train_pipeline

    def dump_model(self, pipelines: dict):
        # Create directory
        path = Path(os.path.dirname(self.config.out_folder))
        path.mkdir(parents=True, exist_ok=True)

        # Save the config file
        with open(f"{self.config.out_folder}/config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)

        # Save the pipelines
        for pipeline in pipelines.keys():
            pickle.dump(
                pipelines[pipeline], open(f"{self.config.out_folder}/{pipeline}", "wb")
            )
        return

    def train(self, dump=True):
        logger.info(
            f"Training Dialogue Act Tagger for {self.config.taxonomy} taxonomy, using the following corpora:"
            f"{[c.name for c in self.corpora]}"
        )
        dataset = []
        pipelines = {}

        for corpus in self.corpora:
            dataset = dataset + corpus.get_train_split()
        if self.config.taxonomy.value.get_dimension_taxonomy() is not None:
            # Train dimension tagger
            logger.info("Training dimension pipeline")
            dimension_dataset = stringify_tags(dataset, "dimension")

            # Filter out unknown tags
            dimension_dataset = [
                u for u in dimension_dataset if all(t > 0 for t in u.tags)
            ]

            pipelines["dimension"] = self.train_pipeline(self.config, dimension_dataset)
            # Train a comm-function classifier for each dimension
            dimension_labels = [[tag for tag in utt.tags] for utt in dimension_dataset]
            dimension_values = list(
                set([label for tagset in dimension_labels for label in tagset])
            )
            for dimension_value in dimension_values:
                logger.info(
                    f"Training communication function pipeline for dimension {dimension_value}"
                )
                comm_dataset = stringify_tags(
                    dataset,
                    "comm_function",
                    filter_attr="dimension",
                    filter_value=dimension_value,
                )

                # Filter out unknown tags
                comm_dataset = [u for u in comm_dataset if all(t > 0 for t in u.tags)]

                pipelines[f"comm_{dimension_value}"] = self.train_pipeline(
                    self.config, comm_dataset
                )
        else:
            logger.info("Training unified communication function pipeline")
            comm_dataset = stringify_tags(dataset, "comm_function")
            pipelines["comm_all"] = self.train_pipeline(self.config, comm_dataset)
        self.config.pipeline_files = list(pipelines.keys())
        if dump:
            self.dump_model(pipelines)
        return SVMTagger(self.config)

    def get_da_distribution(self):
        da_distribution = {"dimension": {}, "comm_function": {}}
        dataset = []
        for c in self.corpora:
            dataset = dataset + c.utterances["train"]
        for dim_value in self.taxonomy.value.get_dimension_taxonomy():
            if dim_value.value > 0:
                da_distribution["dimension"][dim_value] = [
                    u for u in dataset if any(t.dimension == dim_value for t in u.tags)
                ]
                da_distribution["comm_function"][dim_value] = {}
                for comm_value in self.taxonomy.value.get_comm_taxonomy_given_dimension(
                    dim_value.value
                ):
                    da_distribution["comm_function"][dim_value][comm_value] = [
                        u
                        for u in dataset
                        if any(
                            t.dimension == dim_value and t.comm_function == comm_value
                            for t in u.tags
                        )
                    ]
        return da_distribution

    def da_distribution_report(self):
        da_distribution = self.get_da_distribution()
        for k in da_distribution["dimension"]:
            print(f"{k}: {len(da_distribution['dimension'][k])}")
            for c in da_distribution["comm_function"][k]:
                print(f"{c}: {len(da_distribution['comm_function'][k][c])}")
            print("")
