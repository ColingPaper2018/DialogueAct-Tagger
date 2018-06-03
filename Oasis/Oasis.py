import os
import csv
from lxml import etree
from Corpus import Corpus

"""
Oasis class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class Oasis(Corpus):
    def __init__(self, oasis_folder):
        Corpus.__init__(self, oasis_folder)
        self.oasis_folder = oasis_folder
        self.load_csv()

    def load_csv(self):
        # check whether the oasis_folder contains a valid Oasis installation
        try:
            assert os.path.exists(self.oasis_folder)  # folder exists
            assert os.path.exists(self.oasis_folder + "/Data/Lancs_BT150")  # dialogs folders exist
            assert os.path.exists(self.oasis_folder + "/Data/Lancs_BT150/075812009.a.lturn.xml")  # DA files exist
        except AssertionError:
            print("[WARNING] The folder " + self.oasis_folder + " does not contain some important files for the corpus")
            print("Check http://groups.inf.ed.ac.uk/oasis/ for info on how to obtain the complete SWDA corpus.")
            print("")
            self.csv_corpus = None
            return

        dialogs = self.create_dialogs()
        self.csv_corpus = self.create_csv(dialogs)

    def create_dialogs(self):
        dialogs = {}
        for fname in os.listdir(self.oasis_folder + "/Data/Lancs_BT150/"):
            f = open(self.oasis_folder + "/Data/Lancs_BT150/" + fname.strip())
            t = etree.parse(f)
            turns = t.xpath("//lturn")
            for turn in turns:
                self.parse_xml_turn(dialogs, turn)
        return dialogs

    def parse_xml_turn(self, dialogs, turn):
        dialog_id = turn.attrib["id"].split(".")[0]
        try:  # subturn
            turn_id = int(turn.attrib["id"].split(".")[-2])
        except:  # turn
            turn_id = int(turn.attrib["id"].split(".")[-1])
        if dialogs.get(dialog_id, None) is None:  # new dialog
            dialogs[dialog_id] = {}
        if dialogs[dialog_id].get(turn_id, None) is None:  # new turn
            dialogs[dialog_id][turn_id] = []
        segments = turn.xpath(".//segment")
        for segment in segments:
            self.add_segment_to_dialog(dialogs, dialog_id, turn_id, segment)

    def add_segment_to_dialog(self, dialogs, dialog_id, turn_id, segment):
        segm_type = segment.attrib["type"]
        tag = segment.attrib["sp-act"]
        try:
            wFile = segment[0].attrib["href"].split("#")[0]
        except IndexError:
            return
        ids = segment[0].attrib["href"].split("#")[1]
        start_id = ids.split("(")[1].split(")")[0]
        stop_id = ids.split("(")[-1][:-1]
        start_n = int(start_id.split(".")[3])
        text = wFile.split(".xml")[0]
        if not 'anchor' in stop_id:
            stop_n = int(stop_id.split(".")[3])
        else:
            stop_n = start_n
        id_set = ["@id = '" + text + "." + str(i) + "'" for i in range(start_n, stop_n + 1)]
        with open(self.oasis_folder + "/Data/Lancs_BT150/" + wFile) as f:
            tree = etree.parse(f)
            segment = tree.xpath('//*[' + " or ".join(id_set) + ']')
            sentence = " ".join([x.text for x in segment if
                                 x.text is not None and x.text not in ["?", ",", ".", "!", ";"]])
            if sentence != "":
                dialogs[dialog_id][turn_id].append((sentence, tag, segm_type))

    def create_csv(self, dialogs):
        csv_corpus = []
        for d in dialogs:
            prevTag = "other"
            prevType = "other"
            for segm in sorted(dialogs[d].keys()):
                for sentence in dialogs[d][segm]:
                    csv_corpus.append((sentence[0], sentence[1], prevTag, segm, sentence[2], prevType))
                try:
                    prevTag = dialogs[d][segm][-1][1]
                    prevType = dialogs[d][segm][-1][2]
                except IndexError:  # no prev in this segment
                    pass
        return csv_corpus

    @staticmethod
    def da_to_dimension(corpus_tuple):
        da = corpus_tuple[1]
        da_type = corpus_tuple[4]
        if da in ["suggest", "inform", "offer"] or da_type in ["q_wh", "q_yn", "imp"]:
            return "Task"
        elif da in ["thank", "bye", "greet", "pardon", "regret"]:
            return "SocialObligationManagement"
        elif da == "ackn" or da_type == "backchannel":
            return "Feedback"
        else:
            return None

    @staticmethod
    def da_to_cf(corpus_tuple):
        da = corpus_tuple[1]
        da_type = corpus_tuple[4]
        if da_type == "q_wh":
            return "SetQ"
        elif da_type == "q_yn":
            return "CheckQ"
        elif da_type == "imp" or da == "suggest":
            return "Directive"
        elif da == "inform":
            return "Statement"
        elif da == "offer":
            return "Commissive"
        elif da == "thank":
            return "Thanking"
        elif da in ["bye", "greet"]:
            return "Salutation"
        elif da in ["pardon", "regret"]:
            return "Apology"
        elif da == "ackn" or da_type == "backchannel":
            return "Feedback"
        else:
            return None
