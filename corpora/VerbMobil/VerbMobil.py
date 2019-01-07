import os
from corpora.Corpus import Corpus

"""
VerbMobil class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class VerbMobil(Corpus):
    def __init__(self, verbmobil_folder, en_files="files.txt"):
        Corpus.__init__(self, verbmobil_folder)
        # load list of english files
        self.en_files = []
        with open(en_files) as f:
            for line in f:
                self.en_files.append(line.strip())
        self.verbmobil_folder = verbmobil_folder
        self.load_csv()

    def get_corpus_name(self):
        return "VerbMobil II"

    def load_csv(self):
        # check whether the verbmobil_folder contains a valid VerbMobil2 installation
        try:
            assert os.path.exists(self.verbmobil_folder)  # folder exists
            assert os.path.exists(self.verbmobil_folder + "/e001a")  # dialog folders exist
            assert os.path.exists(self.verbmobil_folder + "/e001a/e001ach1_001_SMA.par")  # DA files exist
        except AssertionError:
            print("[WARNING] The folder " + self.verbmobil_folder + " does not contain some important files.")
            print("Look at https://www.phonetik.uni-muenchen.de/Bas/BasVM2eng.html for more information")
            print("")
            self.csv_corpus = None
            return

        raw_data = self.load_raw_data()
        self.csv_corpus = self.create_csv(raw_data)

    def load_raw_data(self):
        data = []
        for en_file in self.en_files:
            segm = 0
            orts = {}
            f = open(self.verbmobil_folder + "/" + en_file, "r")
            for line in f:
                line = line.strip()
                t = line.split(":")[0]
                if t == "ORT":
                    index = line.split("\t")[1]
                    word = line.split("\t")[2]
                    orts[index] = word
                elif t == "DAS":
                    ids = line.split("\t")[0].split(" ")[1].split(",")
                    tag = line.split("\t")[1]
                    sent = ""
                    for id in ids:
                        if not "<" in orts[id]:
                            sent += " " + orts[id].lower()
                    data.append((sent, tag, segm))
                    segm += 1
        return data

    def create_csv(self, raw_data):
        csv_corpus = []
        utt_to_da = {}
        for d in raw_data:
            utt_to_da[d[0][1:]] = (d[1], d[2])
        prev = "@(INIT BA)"
        for utt in utt_to_da:
            csv_corpus.append((utt, utt_to_da[utt][0], prev, utt_to_da[utt][1], None, None))
            prev = utt_to_da[utt][0]
        return csv_corpus

    @staticmethod
    def da_to_dimension(corpus_tuple):
        da = corpus_tuple[1]
        if "FEEDBACK" in da:
            return "Feedback"
        elif any(d in da for d in ["GREET", "BYE", "THANK"]):
            return "SocialObligationManagement"
        elif any(d in da for d in ["OFFER", "SUGGEST", "REQUEST", "INFORM"]):
            return "Task"
        else:
            return None

    @staticmethod
    def da_to_cf(corpus_tuple):
        da = corpus_tuple[1]
        if "OFFER" in da:
            return "Commissive"
        # elif any(d in da for d in ["SUGGEST", "REQUEST"]):
        #    return "Directive"
        elif any(d in da for d in ["GREET", "BYE"]):
            return "Salutation"
        elif "THANK" in da:
            return "Thanking"
        elif "FEEDBACK" in da:
            return "Feedback"
        else:
            return None
