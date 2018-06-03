import os
import re
from Corpus import Corpus

"""
Switchboard class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class Switchboard(Corpus):
    def __init__(self, switchboard_folder, estimator):
        Corpus.__init__(self, switchboard_folder)
        self.switchboard_folder = switchboard_folder
        self.estimator = estimator
        self.load_csv()

    def load_csv(self):
        # check whether the Switchboard contains a valid AMI installation
        try:
            assert os.path.exists(self.switchboard_folder)  # folder exists
            assert os.path.exists(self.switchboard_folder + "/sw00utt")  # dialogs folders exist
            assert os.path.exists(self.switchboard_folder + "/sw00utt/sw_0001_4325.utt")  # DA files exist
        except AssertionError:
            print("[WARNING] The folder " + self.switchboard_folder + " does not contain some important files.")
            print("Check https://catalog.ldc.upenn.edu/ldc97s62 for info on how to obtain the complete SWDA corpus.")
            print("")
            self.csv_corpus = None
            return
        # Read dialogue files from Switchboard
        filelist = self.create_filelist()
        self.csv_corpus = self.create_csv(filelist)

    def create_filelist(self):
        filelist = []
        for folder in os.listdir(self.switchboard_folder):
            if folder.startswith("sw"):  # dialog folder
                for filename in os.listdir(self.switchboard_folder + "/" + folder):
                    if filename.startswith("sw"):  # dialog file
                        filelist.append(self.switchboard_folder + "/" + folder + "/" + filename)
        return filelist

    def create_csv(self, filelist):
        csv_corpus = []
        for filename in filelist:
            prev_speaker = None
            segment = 0
            prev_DAs = {"A": "%", "B": "%"}
            with open(filename) as f:
                utterances = f.readlines()
            for line in utterances:
                line = line.strip()
                try:
                    sentence = line.split("utt")[1].split(":")[1]
                    sw_tag = line.split("utt")[0].split()[0]
                    if "A" in line.split("utt")[0]:  # A speaking
                        speaker = "A"
                    else:
                        speaker = "B"
                except:  # not an SWDA utterance format: probably a header line
                    continue
                if speaker != prev_speaker:
                    prev_speaker = speaker
                    segment += 1
                sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                                  sentence)  # this REGEX removes prosodic information and disfluencies
                sentence = re.sub(r'\W+', ' ', sentence)  # this REGEX removes non alphanumeric characters
                sentence = ' '.join(sentence.split())  # this is just to make extra spaces collapse
                DA_tag = self.estimator.sw_to_damsl(sw_tag, prev_DAs[speaker])
                csv_corpus.append((sentence, DA_tag, prev_DAs[speaker], segment, None, None))
                prev_DAs[speaker] = DA_tag
        return csv_corpus

    @staticmethod
    def da_to_dimension(corpus_tuple):
        da = corpus_tuple[1]
        if da in ["statement-non-opinion", "statement-opinion", "rhetorical-questions", "hedge", "or-clause",
                  "wh-question", "declarative-wh-question", "backchannel-in-question-form", "yes-no-question",
                  "declarative-yn-question", "tag-question", "offers-options-commits", "action-directive"]:
            return "Task"
        elif da in ["thanking", "apology", "downplayer", "conventional-closing"]:
            return "SocialObligationManagement"
        elif da in ["signal-non-understanding", "acknowledge", "appreciation"]:
            return "Feedback"
        else:
            return None

    @staticmethod
    def da_to_cf(corpus_tuple):
        da = corpus_tuple[1]
        if da in ["statement-non-opinion"]:
            return "Statement"
        if da == "or-clause":
            return "ChoiceQ"
        elif da in ["wh-question", "declarative-wh-question"]:
            return "SetQ"
        elif da in ["backchannel-in-question-form", "yes-no-question", "declarative-yn-question", "tag-question"]:
            return "PropQ"
        elif da == "offers-options-commits":
            return "Commissive"
        elif da == "action-directive":
            return "Directive"
        elif da in ["thanking"]:
            return "Thanking"
        elif da in ["apology", "downplayer"]:
            return "Apology"
        elif da in "conventional-closing":
            return "Salutation"
        elif da in ["signal-non-understanding", "acknowledge", "appreciation"]:
            return "Feedback"
        else:
            return None
