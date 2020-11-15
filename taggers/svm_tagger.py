import os
import pickle
from .dialogue_act_tagger import DialogueActTagger
from corpora.taxonomy import Taxonomy, Tag, ISOTag, ISODimension, ISOFeedbackFunction, \
    ISOTaskFunction, ISOSocialFunction, AMITag, AMIFunction
from corpora.Corpus import Utterance
from typing import List
from trainers.svm_trainer import SVMTrainer
from corpora.taxonomy import Layer
from config import Config, SVMConfig


class SVMTagger(DialogueActTagger):
    def __init__(self, cfg: SVMConfig):
        DialogueActTagger.__init__(self, cfg)
        self.acceptance_threshold = cfg.acceptance_threshold
        self.models = {}
        if self.config.taxonomy == Taxonomy.ISO:
            try:
                dimension_file_task = open(os.path.join(cfg.out_folder, "dimension_model_TASK"), "rb")
                self.models['dimension_task'] = pickle.load(dimension_file_task)
                dimension_file_som = open(os.path.join(cfg.out_folder, "dimension_model_SOM"), "rb")
                self.models['dimension_som'] = pickle.load(dimension_file_som)
                dimension_file_fb = open(os.path.join(cfg.out_folder, "dimension_model_FB"), "rb")
                self.models['dimension_fb'] = pickle.load(dimension_file_fb)
                task_file = open(os.path.join(cfg.out_folder, "task_model"), "rb")
                self.models['comm_task'] = pickle.load(task_file)
                som_file = open(os.path.join(cfg.out_folder, "som_model"), "rb")
                self.models['comm_som'] = pickle.load(som_file)
                # fb_file = open(os.path.join(model_folder, "fb_model"))
                # self.fb_model = pickle.load(fb_file)
            except OSError:
                print("The model folder does not contain the required models to run the DA tagger")
                print("Please run the train_all() method of the DialogueActTrain class to obtain the required models")
                exit(1)
        elif self.config.taxonomy == Taxonomy.AMI:
            try:
                ami_file = open(os.path.join(cfg.out_folder, "AMI_model"), "rb")
                self.models['comm_ami'] = pickle.load(ami_file)
            except OSError:
                print("The model folder does not contain the required models to run the DA tagger")
                print("Please run the train_all() method of the DialogueActTrain class to obtain the required models")
                exit(1)

    def tag(self, sentence: Utterance) -> List[Tag]:
        tags = []
        if self.config.taxonomy == Taxonomy.ISO:
            task_dim = self.models['dimension_task'].predict_proba(SVMTrainer.build_features([sentence], self.config,
                                                                                             Layer.Dimension)[0])[0][1]
            som_dim = self.models['dimension_som'].predict_proba(SVMTrainer.build_features([sentence], self.config,
                                                                                           Layer.Dimension)[0])[0][1]
            fb_dim = self.models['dimension_fb'].predict_proba((SVMTrainer.build_features([sentence], self.config,
                                                                                          Layer.Dimension)[0]))[0][0]
            if task_dim > self.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.Task,
                           comm_function=ISOTaskFunction.from_str(self.models['comm_task']))
                )
            if som_dim > self.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.SocialObligation,
                           comm_function=ISOSocialFunction.from_str(self.models['comm_som']))
                )
            if fb_dim > self.acceptance_threshold:
                tags.append(
                    ISOTag(dimension=ISODimension.Feedback,
                           comm_function=ISOFeedbackFunction.Feedback)
                )
        elif self.config.taxonomy == Taxonomy.AMI:
            tags.append(AMITag(comm_function=AMIFunction.from_str(self.models['comm_ami'])))
        return tags
