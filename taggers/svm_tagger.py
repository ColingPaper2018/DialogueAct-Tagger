import os
import pickle
from .dialogue_act_tagger import DialogueActTagger, Classifier
from corpora.taxonomy import Taxonomy, Tag, ISOTag, ISODimension, ISOFeedbackFunction, \
    ISOTaskFunction, ISOSocialFunction, AMITag, AMIFunction
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class SVMTagger(DialogueActTagger):
    @staticmethod
    def from_folder(folder: str) -> "SVMTagger":
        with open(f"{folder}/config.json") as f:
            config = json.load(f)
        return SVMTagger(SVMConfig.from_dict(config))

    def __init__(self, cfg: SVMConfig):
        DialogueActTagger.__init__(self, cfg)
        self.history: List[Utterance] = []
        for pipeline, tagset in self.config.pipeline_files:
            try:
                pipeline_file = open(os.path.join(cfg.out_folder, pipeline), "rb")
                self.classifiers[pipeline] = Classifier(model=pickle.load(pipeline_file), target_tagset=tagset)

            except OSError:
                logging.error("The model folder does not contain the required models to run the DA tagger")
                logging.error("Please run the train_all() method of the "
                              "DialogueActTrain class to obtain the required models")
                exit(1)

    @staticmethod
    def build_features(tagged_utterances: List[Utterance], config: SVMConfig, nlp_inst=None):
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
                features["prev_" + utt.context[0] if len(utt.context) > 0 else "Other"] = True
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

    def tag(self, sentence: Union[Utterance, str]) -> List[Tag]:
        if type(sentence) == str:
            sentence = Utterance(text=sentence, tags=[], context=self.history, speaker_id=0)
        tags = []
        if 'dimension' in self.classifiers:
            features = SVMTagger.build_features([sentence], self.config)[0]
            task_dim = self.classifiers['dimension'].model.predict_proba(features)[0][0]
            som_dim = self.classifiers['dimension'].model.predict_proba(features)[0][1]
            fb_dim = self.classifiers['dimension'].model.predict_proba(features)[0][2]
            if task_dim > self.config.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.Task,
                           comm_function=ISOTaskFunction(self.models['comm_task'].predict(features)[0]))
                )
            if som_dim > self.config.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.SocialObligation,
                           comm_function=ISOSocialFunction(self.models['comm_som'].predict(features)[0]))
                )
            if fb_dim > self.config.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.Feedback,
                           comm_function=ISOFeedbackFunction(self.models['comm_fb'].predict(features)[0]))
                )
        else:
            features = SVMTagger.build_features([sentence], self.config)[0]
            tags.append(self.classifiers['comm_all'].target_tagset(comm_function=AMIFunction(self.models['comm_all'].predict(features)[0])))
        return tags
