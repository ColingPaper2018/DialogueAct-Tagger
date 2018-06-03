import os, sys
import nltk
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.calibration import CalibratedClassifierCV
from Corpus import Corpus
from ItemSelector import ItemSelector
from AMI.AMI import AMI
from VerbMobil.VerbMobil import VerbMobil
from Switchboard.Switchboard import Switchboard
from Switchboard.DAMSL import DAMSL
from Maptask.Maptask import Maptask
from Oasis.Oasis import Oasis
import argparse


class DialogueActTrain:
    def __init__(self, corpora_list):
        self.corpora = []
        for corpus in corpora_list:
            try:
                assert (issubclass(type(corpus), Corpus))
            except AssertionError:
                print("DialogueActTrain error - The corpora list contains objects which are not corpora")
                print("Please ensure each of your corpora is a subclass of Corpus")
                exit(1)
            if corpus.csv_corpus is not None:  # corpus loaded successfully
                self.corpora.append(corpus)
        if len(self.corpora) == 0:
            print("There are no corpora loaded, and the classifier won't train. Please check README.md for information"
                  "on how to obtain more data")
            exit(1)

    @staticmethod
    def build_features(tagged_utterances, indexed_pos=True, ngrams=True, dep=True, prev=True):
        dimension_features = []
        for utt in tagged_utterances:
            features = {}
            features["word_count"] = utt[0].lower()
            features["labels"] = {}
            for i, pos in enumerate(nltk.pos_tag(nltk.word_tokenize(utt[0]))):
                features["labels"]["pos_" + str(i) + pos[1]] = True
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
        try:
            train_pipeline = Pipeline([
                # Use FeatureUnion to combine the features from wordcount and labels
                ('union', FeatureUnion(
                    transformer_list=[('feature_' + str(i), pipeline) for i, pipeline in enumerate(featureset[1])]
                )),
                # Use a SVC classifier on the combined features
                ('classifier', classifier)
            ])
            train_pipeline.fit(featureset[0], [utt[1] for utt in dataset])
            pickle.dump(train_pipeline, open(out_file, 'wb'))
        except ValueError:
            print(f"Not enough data to train the {out_file} classifier! Please check README.md for more information on"
                  f"how to obtain more data")

    def train_all(self, output_folder):
        try:
            assert (os.path.exists(output_folder))
        except AssertionError:
            print(f"folder {output_folder} does not exist: creating it now")
            os.mkdir(output_folder)
        # 1 - generate dimension classifier - task
        dimension_dataset = []
        for corpus in self.corpora:
            dimension_dataset.extend(corpus.dump_iso_dimension_task_csv())
        self.train_classifier(dimension_dataset,
                              self.build_features(dimension_dataset),
                              os.path.join(output_folder, "dimension_model_TASK"),
                              CalibratedClassifierCV(LinearSVC(C=0.1))
                              )
        # 2 - generate dimension classifier - som
        dimension_dataset = []
        for corpus in self.corpora:
            dimension_dataset.extend(corpus.dump_iso_dimension_som_csv())
        self.train_classifier(dimension_dataset,
                              self.build_features(dimension_dataset),
                              os.path.join(output_folder, "dimension_model_SOM"),
                              CalibratedClassifierCV(LinearSVC(C=0.1))
                              )
        # 3 - generate dimension classifier - fb
        dimension_dataset = []
        for corpus in self.corpora:
            dimension_dataset.extend(corpus.dump_iso_dimension_fb_csv())
        self.train_classifier(dimension_dataset,
                              self.build_features(dimension_dataset),
                              os.path.join(output_folder, "dimension_model_FB"),
                              CalibratedClassifierCV(LinearSVC(C=0.1))
                              )
        # 4 - generate task cf classifier
        task_dataset = []
        for corpus in self.corpora:
            task_dataset.extend(corpus.dump_iso_task_csv())
        self.train_classifier(task_dataset,
                              self.build_features(task_dataset),
                              os.path.join(output_folder, "task_model"),
                              CalibratedClassifierCV(LinearSVC(C=0.1))
                              )
        # 5 - generate SOM cf classifier
        som_dataset = []
        for corpus in self.corpora:
            som_dataset.extend(corpus.dump_iso_som_csv())
        self.train_classifier(som_dataset,
                              self.build_features(som_dataset),
                              os.path.join(output_folder, "som_model"),
                              CalibratedClassifierCV(LinearSVC(C=0.1))
                              )
        # 6 - generate FB cf classifier (currently commented due to lack of FB communicative functions
        """
        fb_dataset = []
        for corpus in self.corpora:
            fb_dataset.extend(corpus.dump_iso_fb_csv())
        self.train_classifier(fb_dataset,
                              self.build_fb_features(fb_dataset),
                              os.path.join(output_folder, "fb_model"),
                              LinearSVC()
                              )
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DialogueActTrain - Train a DA Tagger using ISO-converted corpora')

    parser.add_argument('-out-folder', dest='out', type=str,
                        help='where the model files will be stored')
    args = parser.parse_args()
    if args.out is None:
        parser.print_help(sys.stderr)
        exit(1)
    d = DialogueActTrain([Oasis(oasis_folder="Oasis/corpus"),
                          AMI(ami_folder="AMI/corpus"),
                          VerbMobil(verbmobil_folder="VerbMobil/VM2", en_files="VerbMobil/files.txt"),
                          Switchboard(switchboard_folder="Switchboard/SWDA", estimator=DAMSL),
                          Maptask(maptask_folder="Maptask/maptaskv2-1")
                          ])
    d.train_all(args.out)
