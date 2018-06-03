import os
from collections import OrderedDict
from Corpus import Corpus

"""
Maptask class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class Maptask(Corpus):
    def __init__(self, maptask_folder):
        Corpus.__init__(self, maptask_folder)
        self.maptask_folder = maptask_folder
        self.load_csv()

    def load_csv(self):
        # check whether the ami_folder contains a valid AMI installation
        try:
            assert os.path.exists(self.maptask_folder)  # folder exists
            assert os.path.exists(self.maptask_folder + "/Data")  # Data folder exist
            assert os.path.exists(self.maptask_folder + "/Data/timed-units/")  # timed-units folder exists
            assert os.path.exists(self.maptask_folder + "/Data/timed-units/q1ec1.f.timed-units.xml")  # DA files exist
        except AssertionError:
            print("[WARNING] The folder " + self.maptask_folder + " does not contain some files from the corpus.")
            print("You can download a complete version at http://groups.inf.ed.ac.uk/maptask/maptasknxt.html")
            print("")
            self.csv_corpus = None
            return

        dialogs = self.load_dialogs()
        dialogs = self.update_dialogs_with_DAs(dialogs)
        dialogs = self.fix_moves(dialogs)
        self.csv_corpus = self.create_csv(dialogs)

    @staticmethod
    def fix_moves(dialogs):
        fixed_moves = []
        current_move = 2
        current_sentence = ""
        for m in dialogs:
            try:
                move = int(dialogs[m]["move"])
                current_da = dialogs[m]["da"]
            except ValueError:  # transcription error: is not an integer
                continue
            except KeyError:  # no DA for this move
                continue
            if move != current_move:  # add prev move to the fixed_moves array
                fixed_moves.append((current_sentence.strip(), current_da, current_move))
                current_sentence = ""
            current_sentence += (" " + dialogs[m]["text"])
            current_move = int(dialogs[m]["move"])
        return fixed_moves

    def update_dialogs_with_DAs(self, dialogs):
        base_folder = self.maptask_folder + "/Data/moves/"
        for filename in os.listdir(base_folder):
            f = open(base_folder + filename)
            for line in f:
                if "label" not in line:  # not a label: skip line
                    continue
                label = line.split("label=")[1].split("\"")[1].split("\"")[0]
                references = line.split("href")[1].split("\"")[1]
                # 2. Get the filename of the XML file
                fname, ids = references.split("#")
                # 3. Get the range of IDs to be queried
                start_id = ids.split("..")[0].split("(")[1].split(")")[0]
                try:
                    stop_id = ids.split("..")[1].split("(")[1].split(")")[0]
                except:
                    stop_id = start_id
                start_n = int(start_id.split(".")[1])
                text = start_id.split(".")[0]
                stop_n = int(stop_id.split(".")[1])
                set_id = [text + "." + str(i) for i in range(start_n, stop_n + 1)]
                for moveid in set_id:
                    try:
                        dialogs[moveid]["da"] = label
                    except KeyError:  # move has no DA
                        pass
        return dialogs

    def load_dialogs(self):
        base_folder = self.maptask_folder + "/Data/timed-units/"
        dialogs = OrderedDict()
        for filename in os.listdir(base_folder):
            f = open(base_folder + filename)
            speaker = filename.split(".")[1]
            for line in f:
                if "utt" in line:  # utterance
                    move_id = line.split("id=")[1].split("\"")[1].split("\"")[0]
                    utt_id = line.split("utt=")[1].split("\"")[1].split("\"")[0]
                    value = line.split(">")[1].split("<")[0]
                    dialogs[move_id] = {"move": utt_id, "text": value, "speaker": speaker}
        return dialogs

    def create_csv(self, dialogs):
        csv_corpus = []
        conversations = []
        current_move = 0
        current_conv = {}
        for sent, da, move in dialogs:
            if move < current_move and move % 2 == 0:
                conversations.append(current_conv)
                current_conv = {}
            current_move = move
            current_conv[move] = (sent, da)
        for c in conversations:
            segment = 0
            prevDA = "unclassifiable"
            for k in sorted(c.keys()):
                csv_corpus.append(tuple(list(c[k]) + [prevDA, segment, None, None]))
                prevDA = c[k][1]
                segment += 1
        return csv_corpus

    @staticmethod
    def da_to_dimension(corpus_tuple):
        da = corpus_tuple[1]
        if da == "acknowledge":
            return "Feedback"
        elif da == "unclassifiable":
            return None
        else:  # everything else is a task. that's why they call it maptask :)
            return "Task"

    @staticmethod
    def da_to_cf(corpus_tuple):
        da = corpus_tuple[1]
        if da == "acknowledge":
            return "Feedback"
        elif da in ["explain", "clarify", "reply-y", "reply-n", "reply-w"]:
            return "Statement"
        elif da in ["query_yn"]:
            return "PropQ"
        elif da in ["query_w"]:
            return "SetQ"
        elif da in ["instruct"]:
            return "Directive"
        else:
            return None
