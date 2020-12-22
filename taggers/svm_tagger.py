import os
import pickle
from .dialogue_act_tagger import DialogueActTagger
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

from ItemSelector import ItemSelector


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
        self.acceptance_threshold = cfg.acceptance_threshold
        self.models = {}
        self.history: List[Utterance] = []
        for pipeline in self.config.pipeline_files:
            try:
                pipeline_file = open(os.path.join(cfg.out_folder, pipeline), "rb")
                self.models[pipeline] = pickle.load(pipeline_file)

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

    @staticmethod
    def stringify_tags(dataset: List[Utterance], attribute: str, filter_attr: Optional[str] = None,
                       filter_value: Optional[str] = None):
        stringified_dataset = []
        for utterance in dataset:
            new_tags = []
            new_context = []
            for tag in utterance.tags:
                if filter_value is None or getattr(tag, filter_attr).__str__() == filter_value:
                    new_tags.append(getattr(tag, attribute).__str__())
            for tag in utterance.context[0].tags:
                if filter_value is None or getattr(tag, filter_attr).__str__() == filter_value:
                    new_context.append(getattr(tag, attribute).__str__())
            if len(new_tags) > 0:
                stringified_dataset.append(Utterance(
                    speaker_id=utterance.speaker_id,
                    tags=new_tags,
                    context=new_context,
                    text=utterance.text
                ))
        return stringified_dataset

    def tag(self, sentence: Union[Utterance, str]) -> List[Tag]:
        if type(sentence) == str:
            sentence = Utterance(text=sentence, tags=[], context=self.history, speaker_id=0)
        tags = []
        if self.config.taxonomy == Taxonomy.ISO:
            features = SVMTagger.build_features([sentence], self.config)[0]
            task_dim = self.models['dimension_task'].predict_proba(features)[0][1]
            som_dim = self.models['dimension_som'].predict_proba(features)[0][1]
            fb_dim = self.models['dimension_fb'].predict_proba((features)[0])[0][0]
            if task_dim > self.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.Task,
                           comm_function=ISOTaskFunction(self.models['comm_task'].predict(features)[0]))
                )
            if som_dim > self.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.SocialObligation,
                           comm_function=ISOSocialFunction(self.models['comm_som'].predict(features)[0]))
                )
            if fb_dim > self.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.Feedback,
                           comm_function=ISOFeedbackFunction(self.models['comm_fb'].predict(features)[0]))
                )
        elif self.config.taxonomy == Taxonomy.AMI:
            features = SVMTagger.build_features([sentence], self.config)[0]
            tags.append(AMITag(comm_function=AMIFunction(self.models['comm_all'].predict(features)[0])))
        return tags
