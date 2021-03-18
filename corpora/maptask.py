import os
from collections import OrderedDict
from corpora.corpus import Corpus, Utterance
from corpora.taxonomy import Taxonomy, ISOTag, ISODimension, ISOTaskFunction, ISOFeedbackFunction, MaptaskFunction, \
    MaptaskTag
import logging
from typing import List, Dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")
"""
Maptask class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class Maptask(Corpus):
    def __init__(self, maptask_folder, taxonomy: Taxonomy):
        Corpus.__init__(self, "Maptask", maptask_folder, taxonomy)
        self.test_units = ["q1ec3", "q1ec6", "q1ec8", "q1nc4", "q2ec4", "q3ec4", "q4ec3", "q5ec1", "q5ec4",
                           "q5nc5", "q5nc8", "q6ec3", "q6nc3", "q7ec2", "q7ec4", "q7nc7", "q8nc4", "q8nc5"]
        corpus = self.load_corpus(maptask_folder)
        self.utterances = self.parse_corpus(corpus)

    def validate_corpus(self, folder):
        return (os.path.exists(folder)
                and os.path.exists(f"{folder}/Data")
                and os.path.exists(f"{folder}/Data/timed-units")
                and os.path.exists(f"{folder}/Data/timed-units/q1ec1.f.timed-units.xml")
                )

    def load_corpus(self, folder):
        try:
            assert self.validate_corpus(folder)
        except AssertionError:
            logger.warning(f" The folder {folder} does not contain some files from the corpus.")
            logger.info("You can download a complete version at http://groups.inf.ed.ac.uk/maptask/maptasknxt.html")
            return []
        time_unit_folder = f"{folder}/Data/timed-units/"
        dialogs = {"train": OrderedDict(), "test": OrderedDict()}
        for filename in os.listdir(time_unit_folder):
            if any(t in filename for t in self.test_units):
                destination = "test"
            else:
                destination = "train"
            f = open(time_unit_folder + filename)
            speaker = filename.split(".")[1]
            for line in f:
                if "utt" in line:  # utterance
                    move_id = line.split("id=")[1].split("\"")[1].split("\"")[0]
                    utt_id = line.split("utt=")[1].split("\"")[1].split("\"")[0]
                    value = line.split(">")[1].split("<")[0]
                    dialogs[destination][move_id] = {"move": utt_id, "text": value, "speaker": speaker}
        move_folder = f"{folder}/Data/moves/"
        for filename in os.listdir(move_folder):
            f = open(f"{move_folder}/{filename}")
            if any(t in filename for t in self.test_units):
                destination = "test"
            else:
                destination = "train"
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
                        dialogs[destination][moveid]["da"] = label
                    except KeyError:  # move has no DA
                        pass
        fixed_moves = {"train": [], "test": []}
        current_move = 2
        current_sentence = ""
        for destination in dialogs:
            for m in dialogs[destination]:
                try:
                    move = int(dialogs[destination][m]["move"])
                    current_da = dialogs[destination][m]["da"]
                except ValueError:  # transcription error: is not an integer
                    continue
                except KeyError:  # no DA for this move
                    continue
                if move != current_move:  # add prev move to the fixed_moves array
                    fixed_moves[destination].append((current_sentence.strip(), current_da, current_move))
                    current_sentence = ""
                current_sentence += (" " + dialogs[destination][m]["text"])
                current_move = int(dialogs[destination][m]["move"])

        conversations = {"train": [], "test": []}
        for destination in ["train", "test"]:
            current_move = 0
            current_conv = {}
            for sent, da, move in fixed_moves[destination]:
                if move < current_move and move % 2 == 0:
                    conversations[destination].append(current_conv)
                    current_conv = {}
                current_move = move
                current_conv[move] = (sent, da)
        return conversations

    def parse_corpus(self, conversations) -> Dict[str, List[Utterance]]:
        csv_corpus = {"train": [], "test": []}
        for destination in ["train", "test"]:
            for c in conversations[destination]:
                segment = 0
                prev_tags = self.da_to_taxonomy("unk", self.taxonomy)
                prev_sentence = ""
                for k in sorted(c.keys()):
                    sentence, tag = c[k][0], c[k][1]
                    tags = self.da_to_taxonomy(tag, self.taxonomy)
                    if tags[0].comm_function != self.da_to_taxonomy("unk", self.taxonomy):
                        csv_corpus[destination].append(Utterance(text=sentence, tags=tags,
                                                                 speaker_id=segment % 2,
                                                                 context=[Utterance(text=prev_sentence,
                                                                                    tags=prev_tags,
                                                                                    speaker_id=1 - (segment % 2),
                                                                                    context=[])
                                                                          ]))
                    prev_tags = tags
                    prev_sentence = sentence
                    segment += 1
        return csv_corpus

    @staticmethod
    def da_to_taxonomy(dialogue_act: str, taxonomy: Taxonomy, context=None):
        if taxonomy == Taxonomy.ISO:
            if dialogue_act == "acknowledge":
                return [ISOTag(dimension=ISODimension.Feedback, comm_function=ISOFeedbackFunction.Feedback)]
            elif dialogue_act in ["explain", "clarify", "reply-y", "reply-n", "reply-w"]:
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement)]
            elif dialogue_act == "instruct":
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Directive)]
            elif dialogue_act == "query_w":
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.SetQ)]
            elif dialogue_act == "query_yn":
                return [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.PropQ)]
            else:
                return [ISOTag(dimension=ISODimension.Unknown, comm_function=ISOTaskFunction.Unknown)]
        elif taxonomy == Taxonomy.Maptask:
            if dialogue_act == "acknowledge":
                return [MaptaskTag(comm_function=MaptaskFunction.Acknowledge)]
            elif dialogue_act == "align":
                return [MaptaskTag(comm_function=MaptaskFunction.Align)]
            elif dialogue_act == "check":
                return [MaptaskTag(comm_function=MaptaskFunction.Check)]
            elif dialogue_act == "clarify":
                return [MaptaskTag(comm_function=MaptaskFunction.Clarify)]
            elif dialogue_act == "explain":
                return [MaptaskTag(comm_function=MaptaskFunction.Explain)]
            elif dialogue_act == "instruct":
                return [MaptaskTag(comm_function=MaptaskFunction.Instruct)]
            elif dialogue_act == "query-w":
                return [MaptaskTag(comm_function=MaptaskFunction.QueryW)]
            elif dialogue_act == "query-yn":
                return [MaptaskTag(comm_function=MaptaskFunction.QueryYN)]
            elif dialogue_act == "reply-n":
                return [MaptaskTag(comm_function=MaptaskFunction.ReplyN)]
            elif dialogue_act == "reply-w":
                return [MaptaskTag(comm_function=MaptaskFunction.ReplyW)]
            elif dialogue_act == "reply-y":
                return [MaptaskTag(comm_function=MaptaskFunction.Replyy)]
            else:
                return [MaptaskTag(comm_function=MaptaskFunction.Unknown)]
        else:
            raise NotImplementedError(f"Taxonomy {taxonomy} unsupported")
