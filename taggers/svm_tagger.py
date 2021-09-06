import os
import pickle
from .dialogue_act_tagger import DialogueActTagger, Classifier
from corpora.taxonomy import Tag
from corpora.corpus import Utterance
from typing import List
from config import SVMConfig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import spacy
from sklearn.pipeline import Pipeline
from typing import Optional, Union
import json
import logging
from utils import ItemSelector
import numpy as np
from corpora.taxonomy import ISOTag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class DummyClassifier:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return [0.0] * len(X)


class SVMTagger(DialogueActTagger):
    @staticmethod
    def from_folder(folder: str) -> "SVMTagger":
        with open(f"{folder}/config.json") as f:
            config = json.load(f)
        return SVMTagger(SVMConfig.from_dict(config))

    def __init__(self, cfg: SVMConfig):
        DialogueActTagger.__init__(self, cfg)
        self.history: List[Utterance] = []
        for pipeline in self.config.pipeline_files:
            try:
                pipeline_file = open(os.path.join(cfg.out_folder, pipeline), "rb")
                if "dimension" in pipeline:
                    target_tagset = cfg.taxonomy.value.get_dimension_taxonomy()
                elif pipeline == "comm_all":
                    target_tagset = (
                        cfg.taxonomy.value.get_comm_taxonomy_given_dimension()
                    )
                else:
                    target_tagset = (
                        cfg.taxonomy.value.get_comm_taxonomy_given_dimension(
                            int(pipeline.split("comm_")[1])
                        )
                    )
                self.classifiers[pipeline] = Classifier(
                    model=pickle.load(pipeline_file), target_tagset=target_tagset
                )
            except OSError:
                logging.error(
                    "The model folder does not contain the required models to run the DA tagger"
                )
                logging.error(
                    "Please run the train_all() method of the "
                    "DialogueActTrain class to obtain the required models"
                )
                exit(1)
        self.nlp_inst = spacy.load("en")

    @staticmethod
    def build_features(
        tagged_utterances: List[Utterance], config: SVMConfig, nlp_inst=None
    ):
        if nlp_inst is None:
            nlp_inst = spacy.load("en")
        dimension_features = []
        docs = nlp_inst.pipe([utt.text for utt in tagged_utterances])
        for idx, (utt, doc) in enumerate(zip(tagged_utterances, docs)):
            features = {"word_count": utt.text.lower(), "labels": {}}
            for i, tok in enumerate(doc):
                if config.indexed_pos:
                    features["labels"][f"pos_{tok.tag_}_{str(i)}"] = True
                    features["labels"][f"pos_{tok.tag_}_{tok.text}"] = True
                if config.dep:
                    features["labels"][f"dep_{tok.dep_}"] = True
                if config.pos:
                    features["labels"][f"dep_{tok.tag_}"] = True
                if config.indexed_dep:
                    features["labels"][f"idep_{tok.dep_}_{str(i)}"] = True
            if config.prev:
                features[
                    "prev_" + str(utt.context[0]) if len(utt.context) > 0 else "Other"
                ] = True
            dimension_features.append(features)
        wordcount_pipeline = Pipeline(
            [
                ("selector", ItemSelector(key="word_count")),
                ("vectorizer", CountVectorizer(ngram_range=(1, 2))),
            ]
        )
        label_pipeline = Pipeline(
            [("selector", ItemSelector(key="labels")), ("vectorizer", DictVectorizer())]
        )
        return dimension_features, [wordcount_pipeline, label_pipeline]

    def annotate_features(self, features) -> List[Tag]:
        tags = []
        if "dimension" in self.classifiers:
            prediction = self.classifiers["dimension"].model.predict_proba(features)[0]
            for dimension in range(0, len(prediction)):
                feature_dim = prediction[dimension]
                if feature_dim > self.config.acceptance_threshold:
                    tags.append(
                        self.config.taxonomy.value(
                            dimension=self.classifiers["dimension"].target_tagset(
                                dimension + 1
                            ),
                            comm_function=self.classifiers[
                                f"comm_{dimension + 1}"
                            ].target_tagset(
                                self.classifiers[f"comm_{dimension + 1}"].model.predict(
                                    features
                                )[0]
                            ),
                        )
                    )
        else:
            tags.append(
                self.classifiers["comm_all"].target_tagset(
                    comm_function=self.classifiers["comm_all"].target_tagset(
                        self.classifiers["comm_all"].model.predict(features)[0]
                    )
                )
            )
        return tags

    def tag(self, sentence: Union[Utterance, str]) -> List[Tag]:
        if type(sentence) == str:
            sentence = Utterance(
                text=sentence, tags=[], context=self.history, speaker_id=0
            )
        features = SVMTagger.build_features([sentence], self.config)[0]
        return self.annotate_features(features)

    def tag_batch(self, batch: List[Union[Utterance, str]]) -> List[List[Tag]]:
        batch = [
            Utterance(text=sentence, tags=[], context=self.history, speaker_id=0)
            if type(sentence) == str
            else sentence
            for sentence in batch
        ]
        featurized_batch = self.build_features(batch, self.config, self.nlp_inst)[0]

        tagged_batch = [
            self.annotate_features([features]) for features in featurized_batch
        ]
        return tagged_batch
