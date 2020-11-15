import os
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from taggers.svm_tagger import SVMTagger
from config import SVMConfig
from corpora.Corpus import Utterance
from corpora.taxonomy import Layer, Taxonomy, ISODimension
from typing import List

from ItemSelector import ItemSelector
from .trainer import Trainer
from pathlib import Path
import spacy
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class SVMTrainer(Trainer):
    def __init__(self, corpora_list, config: SVMConfig):
        Trainer.__init__(self, corpora_list, config)

    @staticmethod
    def build_features(tagged_utterances: List[Utterance], config: SVMConfig, layer: Layer,
                       nlp_inst=None):
        if nlp_inst is None:
            nlp_inst = spacy.load("en")
        dimension_features = []
        docs = nlp_inst.pipe([utt.text for utt in tagged_utterances])
        for idx, (utt, doc) in enumerate(zip(tagged_utterances, docs)):
            features = {"word_count": utt.text.lower(), "labels": {}}
            for i, tok in enumerate(doc):
                if config.indexed_pos:
                    features["labels"]["pos_" + tok.tag_] = True
                if config.dep:
                    features["labels"][tok.dep_] = True
                if config.indexed_dep:
                    features["labels"]["pos_" + tok.dep_] = True
            if config.prev:
                features["prev_" + (tagged_utterances[idx - 1].tags[0] if idx > 0 else "Other")] = True
            dimension_features.append(features)
        wordcount_pipeline = Pipeline([
            ('selector', ItemSelector(key='word_count')),
            ('vectorizer', CountVectorizer(ngram_range=(1, 2)))
        ])
        label_pipeline = Pipeline([
            ('selector', ItemSelector(key='labels')),
            ('vectorizer', DictVectorizer())
        ])
        return dimension_features, [wordcount_pipeline, label_pipeline]

    @staticmethod
    def train_pipeline(config: SVMConfig, layer: Layer, dataset: List[Utterance]):
        features = SVMTrainer.build_features(dataset, config, layer=layer)
        train_pipeline = Pipeline([
            # Use FeatureUnion to combine the features from wordcount and labels
            ('union', FeatureUnion(
                transformer_list=[('feature_' + str(i), pipeline) for i, pipeline in enumerate(features[1])]
            )),
            # Use a SVC classifier on the combined features
            ('classifier', config.classifier)
        ])
        if len(dataset) == 0:
            logger.error(f"Not enough data to train the {out_file} classifier! Please check README.md for "
                         f"more information on how to obtain more data")
            return
        train_pipeline.fit(features[0], [utt[1] for utt in dataset])
        return train_pipeline

    def train(self, out_file: str):
        logger.info(f"Training Dialogue Act Tagger for {self.config.taxonomy} taxonomy, using the following corpora:"
                    f"{[c.name for c in self.corpora]}")
        dataset = []
        for corpus in self.corpora:
            dataset = dataset + corpus.utterances

        if self.config.taxonomy == Taxonomy.AMI:
            train_pipeline = self.train_pipeline(self.config, Layer.CommFunction, dataset)
            path = Path(os.path.dirname(out_file))
            path.mkdir(parents=True, exist_ok=True)
            pickle.dump(train_pipeline, open(out_file, 'wb'))
            return SVMTagger(os.path.dirname(out_file))
        elif self.config.taxonomy == Taxonomy.ISO:
            # Train dimension tagger
            dimension_pipeline = self.train_pipeline(self.config, Layer.Dimension, dataset)

            # Train task tagger
            task_dataset = [u for u in dataset if u.tags.dimension == ISODimension.Task]
            task_pipeline = self.train_pipeline(self.config, Layer.CommFunction, task_dataset)

            # Train som tagger
            som_dataset = [u for u in dataset if u.tags.dimension == ISODimension.SocialObligation]
            som_pipeline = self.train_pipeline(self.config, Layer.CommFunction, som_dataset)
            path = Path(os.path.dirname(out_file))
            path.mkdir(parents=True, exist_ok=True)
            pickle.dump(dimension_pipeline, open(f"{out_file}_dim", 'wb'))
            pickle.dump(task_pipeline, open(f"{out_file}_task", 'wb'))
            pickle.dump(som_pipeline, open(f"{out_file}_som", 'wb'))






