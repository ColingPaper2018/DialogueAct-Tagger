import os
import pickle
from .predictor import Predictor
from trainers.svm_trainer import SVMTrain


class SVMPredictor(Predictor):
    def __init__(self, cfg):
        self.acceptance_threshold = cfg.acceptance_threshold
        try:
            dimension_file_task = open(os.path.join(cfg.out_folder, "dimension_model_TASK"), "rb")
            self.dimension_model_task = pickle.load(dimension_file_task)
            dimension_file_som = open(os.path.join(cfg.out_folder, "dimension_model_SOM"), "rb")
            self.dimension_model_som = pickle.load(dimension_file_som)
            dimension_file_fb = open(os.path.join(cfg.out_folder, "dimension_model_FB"), "rb")
            self.dimension_model_fb = pickle.load(dimension_file_fb)
            task_file = open(os.path.join(cfg.out_folder, "task_model"), "rb")
            self.task_model = pickle.load(task_file)
            som_file = open(os.path.join(cfg.out_folder, "som_model"), "rb")
            self.som_model = pickle.load(som_file)
            # fb_file = open(os.path.join(model_folder, "fb_model"))
            # self.fb_model = pickle.load(fb_file)
        except OSError:
            print("The model folder does not contain the required models to run the DA tagger")
            print("Please run the train_all() method of the DialogueActTrain class to obtain the required models")
            exit(1)

    def dialogue_act_tag(self, sentence, prev_da=None):
        if prev_da is None:
            prev_da = "Other"
        da = []
        utt = [sentence, prev_da]
        task_dim = self.dimension_model_task.predict_proba(SVMTrain.build_features([utt])[0])[0][1]
        som_dim = self.dimension_model_som.predict_proba(SVMTrain.build_features([utt])[0])[0][1]
        fb_dim = self.dimension_model_fb.predict_proba((SVMTrain.build_features([utt])[0]))[0][0]
        if task_dim > self.acceptance_threshold:
            da.append(
                {'dimension': "Task",
                 'communicative_function': self.task_model.predict(SVMTrain.build_features([utt])[0])[0]})
        if som_dim > self.acceptance_threshold:
            da.append({'dimension': "SocialObligationManagement",
                       'communicative_function': self.som_model.predict(SVMTrain.build_features([utt])[0])[0]})
        elif fb_dim > self.acceptance_threshold:
            da.append({'dimension': "Feedback", 'communicative_function': "Feedback"})
        return da

    def tag_task(self, sentence, prev_da=None):
        utt = [sentence, prev_da]
        return self.task_model.predict(SVMTrain.build_features([utt])[0])[0]

    def tag_som(self, sentence, prev_da=None):
        utt = [sentence, prev_da]
        return self.som_model.predict(SVMTrain.build_features([utt])[0])[0]