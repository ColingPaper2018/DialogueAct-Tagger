import os
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.calibration import CalibratedClassifierCV
from corpora.Corpus import Corpus
from ItemSelector import ItemSelector
from .trainer import Trainer
from pathlib import Path
import spacy
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


class SVMTrain(Trainer):
    def __init__(self, corpora_list):
        self.corpora = []
        for corpus in corpora_list:
            try:
                assert (issubclass(type(corpus), Corpus))
            except AssertionError:
                logger.error("DialogueActTrain error - The corpora list contains objects which are not corpora")
                logger.error("Please ensure each of your corpora is a subclass of Corpus")
                exit(1)
            if corpus.csv_corpus is not None:  # corpus loaded successfully
                self.corpora.append(corpus)
        if len(self.corpora) == 0:
            logger.error("There are no corpora loaded, and the classifier won't train. "
                         "Please check README.md for information on how to obtain more data")
            exit(1)

    @staticmethod
    def build_features(tagged_utterances, indexed_pos=True, indexed_dep=False,
                       ngrams=True, dep=True, prev=True, nlp_inst=None):
        if nlp_inst is None:
            nlp_inst = spacy.load("en")
        dimension_features = []
        docs = nlp_inst.pipe([utt[0] for utt in tagged_utterances])
        for utt, doc in zip(tagged_utterances, docs):
            features = {"word_count": utt[0].lower(), "labels": {}}
            for i, tok in enumerate(doc):
                if indexed_pos:
                    features["labels"]["pos_" + tok.tag_] = True
                if dep:
                    features["labels"][tok.dep_] = True
                if indexed_dep:
                    features["labels"]["pos_" + tok.dep_] = True
            if prev:
                features["prev_" + utt[1]] = True
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
    def train_classifier(dataset, featureset, out_file, classifier):
        train_pipeline = Pipeline([
            # Use FeatureUnion to combine the features from wordcount and labels
            ('union', FeatureUnion(
                transformer_list=[('feature_' + str(i), pipeline) for i, pipeline in enumerate(featureset[1])]
            )),
            # Use a SVC classifier on the combined features
            ('classifier', classifier)
        ])
        if len(dataset) == 0:
            logger.error(f"Not enough data to train the {out_file} classifier! Please check README.md for "
                         f"more information on how to obtain more data")
            return
        train_pipeline.fit(featureset[0], [utt[1] for utt in dataset])
        path = Path(os.path.dirname(out_file))
        path.mkdir(parents=True, exist_ok=True)
        pickle.dump(train_pipeline, open(out_file, 'wb'))

    def train_som(self, out_file):
        logger.info("Training the SOM communicative function classifier")
        som_dataset = []
        for corpus in self.corpora:
            som_dataset.extend(corpus.dump_iso_som_csv())
        if len(som_dataset) == 0:
            logger.error(f"Not enough data to train the {out_file} classifier! Please check README.md for "
                         f"more information on how to obtain more data")
            return
        self.train_classifier(som_dataset,
                              self.build_features(som_dataset),
                              os.path.join(out_file, "som_model"),
                              CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)
                              )

    def train_task(self, out_file):
        logger.info("Training the task communicative function classifier")
        # 4 - generate task cf classifier
        task_dataset = []
        for corpus in self.corpora:
            task_dataset.extend(corpus.dump_iso_task_csv())
        if len(task_dataset) == 0:
            logger.error(f"Not enough data to train the {out_file} classifier! Please check README.md for "
                         f"more information on how to obtain more data")
            return
        self.train_classifier(task_dataset,
                              self.build_features(task_dataset),
                              os.path.join(out_file, "task_model"),
                              CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)
                              )

    def train_dimension(self, out_file):
        logger.info("Training dimension classifier for Task")
        # 1 - generate dimension classifier - task
        dimension_dataset = []
        for corpus in self.corpora:
            dimension_dataset.extend(corpus.dump_iso_dimension_task_csv())
        self.train_classifier(dimension_dataset,
                              self.build_features(dimension_dataset),
                              os.path.join(out_file, "dimension_model_TASK"),
                              CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)
                              )
        # 2 - generate dimension classifier - som
        logger.info("Training dimension classifier for SOM")
        dimension_dataset = []
        for corpus in self.corpora:
            dimension_dataset.extend(corpus.dump_iso_dimension_som_csv())
        self.train_classifier(dimension_dataset,
                              self.build_features(dimension_dataset),
                              os.path.join(out_file, "dimension_model_SOM"),
                              CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)
                              )
        # 3 - generate dimension classifier - fb
        logger.info("Training dimension classifier for Feedback")
        dimension_dataset = []
        for corpus in self.corpora:
            dimension_dataset.extend(corpus.dump_iso_dimension_fb_csv())
        self.train_classifier(dimension_dataset,
                              self.build_features(dimension_dataset),
                              os.path.join(out_file, "dimension_model_FB"),
                              CalibratedClassifierCV(LinearSVC(C=0.1), cv=3)
                              )

    def train_all(self, output_folder):
        self.train_dimension(output_folder)
        self.train_task(output_folder)
        self.train_som(output_folder)
