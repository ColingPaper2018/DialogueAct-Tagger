import os
from collections import OrderedDict
from corpora.Corpus import Corpus, Utterance
from corpora.taxonomy import ISOTag, ISODimension, ISOTaskFunction, ISOFeedbackFunction, AMIFunction, AMITag

import logging
from corpora.taxonomy import Taxonomy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")

"""
AMI class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class AMI(Corpus):
    def __init__(self, corpus_folder):
        Corpus.__init__(self, corpus_folder)
        corpus = self.load_corpus(corpus_folder)
        self.utterances = self.parse_corpus(corpus, Taxonomy.AMI)

    def validate_corpus(self, corpus_folder):
            return (os.path.exists(corpus_folder + "/words/ES2002a.A.words.xml")  # word files exist
                    and os.path.exists(corpus_folder + "/dialogueActs/ES2002a.A.dialog-act.xml"))  # DA files exist

    def load_corpus(self, corpus_folder):
        # Check that the corpus files are there
        try:
            self.validate_corpus(corpus_folder)
        except AssertionError:
            logger.warning("The folder " + corpus_folder +
                           " does not contain some important files from the corpus.")
            logger.info("You can download a complete version of the corpus at http://groups.inf.ed.ac.uk/ami/download/")
            self.utterances = None
            return

        dialogs = {}  # this will store dialogs from the corpus
        dialog_names = []  # this will store file names from the corpus
        for dialog_name in os.listdir(corpus_folder + "/dialogueActs/"):
            if "dialog-act" in dialog_name:  # DA file
                dialog_names.append(dialog_name.split("dialog-act")[0])
        for dialog_name in dialog_names:
            dialogs[dialog_name] = OrderedDict()
            with open(corpus_folder + "/words/" + dialog_name + "words.xml") as wfile:
                for line in wfile:
                    if "<w" not in line:  # not a word
                        continue
                    elif "punc=" in line:  # punctuation
                        continue
                    word_id = line.split("id=\"")[1].split("\"")[0].split(".")[-1]
                    word_value = line.split(">")[1].split("<")[0]
                    dialogs[dialog_name][word_id] = []
                    dialogs[dialog_name][word_id].append(word_value)
            with open(corpus_folder + "/dialogueActs/" + dialog_name + "dialog-act.xml") as actfile:
                dact = ""
                for line in actfile:
                    if "<nite:pointer" in line:  # act definition
                        dact = line.split("href=\"da-types.xml#id(")[1].split(")")[0]
                        continue
                    elif "<nite:child" in line:  # word list for this act
                        ids = line.split("#")[1]
                        # 4.1 get the start/stopIDs to be queried
                        start_id = ids.split("..")[0].split("(")[1].split(")")[0].split("words")[1]
                        # 4.2 Get the range of IDs to be queried
                        try:
                            stop_id = ids.split("..")[1].split("(")[1].split(")")[0].split("words")[1]
                        except IndexError:
                            stop_id = start_id
                        start_n = int(start_id)
                        stop_n = int(stop_id)
                        # 4. Build the query
                        set = ["words" + str(i) for i in range(start_n, stop_n + 1)]
                        for w in set:
                            try:
                                dialogs[dialog_name][w].append(dact)
                            except KeyError:
                                continue
            with open(corpus_folder + "/segments/" + dialog_name + "segments.xml") as segmfile:
                segment = 0
                for line in segmfile:
                    if "<nite:child" in line:  # word list for this segment
                        ids = line.split("#")[1]
                        # 4.1 get the start/stopIDs to be queried
                        start_id = ids.split("..")[0].split("(")[1].split(")")[0].split("words")[1]
                        # 4.2 Get the range of IDs to be queried
                        try:
                            stop_id = ids.split("..")[1].split("(")[1].split(")")[0].split("words")[1]
                        except IndexError:
                            stop_id = start_id
                        start_n = int(start_id)
                        stop_n = int(stop_id)
                        # 4. Build the query
                        set = ["words" + str(i) for i in range(start_n, stop_n + 1)]
                        for w in set:
                            try:
                                dialogs[dialog_name][w].append(str(segment))
                            except KeyError:
                                continue
                    segment += 1
        return dialogs

    def parse_corpus(self, dialogs, taxonomy: Taxonomy):
        parsed_corpus = []
        for d in dialogs:
            speaker_id = 0
            sentence = ""
            previous_utterance = Utterance(
                text="",
                tags=[],
                context=[],
                speaker_id=speaker_id
            )
            current_segment = -1
            for word in dialogs[d]:
                try:
                    word, DA, segment = dialogs[d][word]
                except ValueError:
                    continue
                parsed_da = self.da_to_taxonomy(DA, taxonomy)
                previous_da = previous_utterance.tags[0] if len(previous_utterance.tags) > 0 else None
                if (parsed_da != previous_da or current_segment != segment) and sentence != "":  # new DA or segment
                    parsed_corpus.append(
                        Utterance(text=sentence.strip().replace("&#39;", "'"),
                                  tags=self.da_to_taxonomy(DA, taxonomy),
                                  context=[previous_utterance],
                                  speaker_id=speaker_id)
                    )
                    previous_utterance = Utterance(text=sentence.strip().replace("&#39;", "'"),
                                                   tags=self.da_to_taxonomy(DA, taxonomy),
                                                   context=[previous_utterance],
                                                   speaker_id=speaker_id)
                    sentence = ""
                current_segment = segment
                sentence = sentence + " " + (word.strip())

        return parsed_corpus

    @staticmethod
    def da_to_taxonomy(dialogue_act: str, taxonomy: Taxonomy, context=None):
        if taxonomy == Taxonomy.ISO:
            if dialogue_act == "ami_da_4":
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement)]
            elif dialogue_act == "ami_da_7":
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Commissive)]
            elif dialogue_act == "ami_da_6":
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Directive)]
            elif dialogue_act in ["ami_da_1", "ami_da_9"]:
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOFeedbackFunction.Feedback)]
            else:
                return None
        elif taxonomy == Taxonomy.AMI:
            if dialogue_act == "ami_da_1":
                return [AMITag(comm_function=AMIFunction.Backchannel)]
            elif dialogue_act == "ami_da_2":
                return [AMITag(comm_function=AMIFunction.Stall)]
            elif dialogue_act == "ami_da_3":
                return [AMITag(comm_function=AMIFunction.Fragment)]
            elif dialogue_act == "ami_da_4":
                return [AMITag(comm_function=AMIFunction.Inform)]
            elif dialogue_act == "ami_da_5":
                return [AMITag(comm_function=AMIFunction.ElicitInform)]
            elif dialogue_act == "ami_da_6":
                return [AMITag(comm_function=AMIFunction.Suggest)]
            elif dialogue_act == "ami_da_7":
                return [AMITag(comm_function=AMIFunction.Offer)]
            elif dialogue_act == "ami_da_8":
                return [AMITag(comm_function=AMIFunction.ElicitOfferOrSuggest)]
            elif dialogue_act == "ami_da_9":
                return [AMITag(comm_function=AMIFunction.Assess)]
            elif dialogue_act == "ami_da_11":
                return [AMITag(comm_function=AMIFunction.ElicitAssess)]
            elif dialogue_act == "ami_da_12":
                return [AMITag(comm_function=AMIFunction.CommentAboutUnderstanding)]
            elif dialogue_act == "ami_da_13":
                return [AMITag(comm_function=AMIFunction.ElicitCommentAboutUnderstanding)]
            elif dialogue_act == "ami_da_14":
                return [AMITag(comm_function=AMIFunction.BePositive)]
            elif dialogue_act == "ami_da_15":
                return [AMITag(comm_function=AMIFunction.BeNegative)]
            elif dialogue_act == "ami_da_16":
                return [AMITag(comm_function=AMIFunction.Other)]
            else:
                return None
