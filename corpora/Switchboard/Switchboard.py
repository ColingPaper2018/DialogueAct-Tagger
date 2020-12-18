import os
import re
from corpora.Corpus import Corpus
import logging
from corpora.taxonomy import Taxonomy, ISOTag, ISODimension, ISOTaskFunction, ISOFeedbackFunction, ISOSocialFunction, SWDAFunction, SWDATag, Tag
from typing import List
from corpora.Corpus import Utterance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


"""
Switchboard class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class Switchboard(Corpus):
    def __init__(self, switchboard_folder, taxonomy: Taxonomy):
        Corpus.__init__(self, "Switchboard", switchboard_folder, taxonomy)
        corpus = self.load_corpus(switchboard_folder)
        self.utterances = self.parse_corpus(corpus)

    def validate_corpus(self, folder):
        return os.path.exists(folder) and os.path.exists(f"{folder}/sw00utt") \
               and os.path.exists(f"{folder}/sw00utt/sw_0001_4325.utt")

    def load_corpus(self, folder):
        # check whether the folder contains a valid Switchboard installation
        try:
            assert self.validate_corpus(folder)
        except AssertionError:
            logging.warning(f"The folder {folder} does not contain some important files.")
            logging.info("Check https://catalog.ldc.upenn.edu/ldc97s62 "
                         "for info on how to obtain the complete SWDA corpus.")
            return
        # Read dialogue files from Switchboard
        filelist = []
        for file_folder in os.listdir(folder):
            if file_folder.startswith("sw"):  # dialog folder
                for filename in os.listdir(folder + "/" + file_folder):
                    if filename.startswith("sw"):  # dialog file
                        filelist.append(folder + "/" + file_folder + "/" + filename)
        conversations = {}
        for filename in filelist:
            with open(filename) as f:
                utterances = f.readlines()
            conversations[filename] = utterances
        return conversations

    def parse_corpus(self, conversations):
        utterances = []
        for filename in conversations.keys():
            prev_speaker = None
            segment = 0
            prev_tags = {0: self.da_to_taxonomy("%", self.taxonomy, []), 1: self.da_to_taxonomy("%", self.taxonomy, [])}
            prev_texts = {0: "", 1: ""}
            for line in conversations[filename]:
                line = line.strip()
                try:
                    sentence = line.split("utt")[1].split(":")[1]
                    sw_tag = line.split("utt")[0].split()[0]
                    if "A" in line.split("utt")[0]:  # A speaking
                        speaker = 0
                    else:
                        speaker = 1
                except IndexError:  # not an SWDA utterance format: probably a header line
                    continue
                if speaker != prev_speaker:
                    prev_speaker = speaker
                    segment += 1
                sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                                  sentence)  # this REGEX removes prosodic information and disfluencies
                sentence = re.sub(r'\W+', ' ', sentence)  # this REGEX removes non alphanumeric characters
                sentence = ' '.join(sentence.split())  # this is just to make extra spaces collapse
                tags = self.da_to_taxonomy(sw_tag, self.taxonomy, prev_tags[speaker])
                utterances.append(Utterance(text=sentence, tags=tags,
                                            context=[Utterance(prev_texts[speaker], prev_tags[speaker], [], speaker)],
                                            speaker_id=speaker))
                prev_tags[speaker] = tags
                prev_texts[speaker] = sentence
        return utterances

    @staticmethod
    def da_to_taxonomy(dialogue_act: str, taxonomy: Taxonomy, context: List[Tag]) -> List[Tag]:
        if dialogue_act == "+":
            new_tag = context
        else:
            if taxonomy == Taxonomy.SWDA:
                mapping_dict = {
                    "sd": [SWDATag(comm_function=SWDAFunction.StatementNonOpinion)],
                    "b":  [SWDATag(comm_function=SWDAFunction.Acknowledge)],
                    "sv": [SWDATag(comm_function=SWDAFunction.StatementOpinion)],
                    "aa": [SWDATag(comm_function=SWDAFunction.AgreeAccept)],
                    "%-": [SWDATag(comm_function=SWDAFunction.Abandoned)],
                    "ba": [SWDATag(comm_function=SWDAFunction.Appreciation)],
                    "qy": [SWDATag(comm_function=SWDAFunction.YesNoQuestion)],
                    "x":  [SWDATag(comm_function=SWDAFunction.NonVerbal)],
                    "ny": [SWDATag(comm_function=SWDAFunction.YesAnswers)],
                    "fc": [SWDATag(comm_function=SWDAFunction.ConventionalClosing)],
                    "%":  [SWDATag(comm_function=SWDAFunction.Uninterpretable)],
                    "qw": [SWDATag(comm_function=SWDAFunction.WhQuestion)],
                    "nn": [SWDATag(comm_function=SWDAFunction.NoAnswers)],
                    "bk": [SWDATag(comm_function=SWDAFunction.ResponseAcknowledgement)],
                    "h":  [SWDATag(comm_function=SWDAFunction.Hedge)],
                    "qy^d": [SWDATag(comm_function=SWDAFunction.DeclarativeYNQuestion)],
                    "o": [SWDATag(comm_function=SWDAFunction.Other)],
                    "bh": [SWDATag(comm_function=SWDAFunction.BackchannelInQuestionForm)],
                    "^q": [SWDATag(comm_function=SWDAFunction.Quotation)],
                    "bf": [SWDATag(comm_function=SWDAFunction.SummarizeReformulate)],
                    "na": [SWDATag(comm_function=SWDAFunction.AffirmativeNonYesAnswers)],
                    "ny^e": [SWDATag(comm_function=SWDAFunction.AffirmativeNonYesAnswers)],
                    "ad": [SWDATag(comm_function=SWDAFunction.ActionDirective)],
                    "^2": [SWDATag(comm_function=SWDAFunction.CollaborativeCompletion)],
                    "b^m": [SWDATag(comm_function=SWDAFunction.RepeatPhrase)],
                    "qo": [SWDATag(comm_function=SWDAFunction.OpenQuestion)],
                    "qh": [SWDATag(comm_function=SWDAFunction.RhetoricalQuestions)],
                    "^h": [SWDATag(comm_function=SWDAFunction.Hold)],
                    "ar": [SWDATag(comm_function=SWDAFunction.Reject)],
                    "ng": [SWDATag(comm_function=SWDAFunction.NegativeNonNoAnswers)],
                    "nn^e": [SWDATag(comm_function=SWDAFunction.NegativeNonNoAnswers)],
                    "br": [SWDATag(comm_function=SWDAFunction.SignalNonUnderstanding)],
                    "no": [SWDATag(comm_function=SWDAFunction.OtherAnswers)],
                    "fp": [SWDATag(comm_function=SWDAFunction.ConventionalOpening)],
                    "qrr": [SWDATag(comm_function=SWDAFunction.OrClause)],
                    "arp": [SWDATag(comm_function=SWDAFunction.DispreferredAnswers)],
                    "nd": [SWDATag(comm_function=SWDAFunction.DispreferredAnswers)],
                    "t3": [SWDATag(comm_function=SWDAFunction.ThirdPartyTalk)],
                    "oo": [SWDATag(comm_function=SWDAFunction.OffersOptionsCommits)],
                    "co": [SWDATag(comm_function=SWDAFunction.OffersOptionsCommits)],
                    "cc": [SWDATag(comm_function=SWDAFunction.OffersOptionsCommits)],
                    "t1": [SWDATag(comm_function=SWDAFunction.SelfTalk)],
                    "bd": [SWDATag(comm_function=SWDAFunction.Downplayer)],
                    "aap": [SWDATag(comm_function=SWDAFunction.MaybeAcceptPart)],
                    "am": [SWDATag(comm_function=SWDAFunction.MaybeAcceptPart)],
                    "^g": [SWDATag(comm_function=SWDAFunction.TagQuestion)],
                    "qw^d": [SWDATag(comm_function=SWDAFunction.DeclarativeWhQuestion)],
                    "fa": [SWDATag(comm_function=SWDAFunction.Apology)],
                    "ft": [SWDATag(comm_function=SWDAFunction.Thanking)],
                }
            elif taxonomy == Taxonomy.ISO:
                mapping_dict = {
                    "sd": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement)],
                    "qrr": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.ChoiceQ)],
                    "qw^d": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.SetQ)],
                    "qw": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.SetQ)],
                    "bh": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.PropQ)],
                    "qy": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.PropQ)],
                    "qy^d": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.PropQ)],
                    "^g": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.PropQ)],
                    "oo": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Commissive)],
                    "co": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Commissive)],
                    "cc": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Commissive)],
                    "ad": [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Directive)],
                    "ft": [ISOTag(dimension=ISODimension.SocialObligation, comm_function=ISOSocialFunction.Thanking)],
                    "fa": [ISOTag(dimension=ISODimension.SocialObligation, comm_function=ISOSocialFunction.Apology)],
                    "bd": [ISOTag(dimension=ISODimension.SocialObligation, comm_function=ISOSocialFunction.Apology)],
                    "fc": [ISOTag(dimension=ISODimension.SocialObligation, comm_function=ISOSocialFunction.Salutation)],
                    "br": [ISOTag(dimension=ISODimension.Feedback, comm_function=ISOFeedbackFunction.Feedback)],
                    "b": [ISOTag(dimension=ISODimension.Feedback, comm_function=ISOFeedbackFunction.Feedback)],
                    "ba": [ISOTag(dimension=ISODimension.Feedback, comm_function=ISOFeedbackFunction.Feedback)],
                    "%":  [ISOTag(dimension=ISODimension.Task, comm_function=ISOTaskFunction.Unknown)]
                }
            else:
                raise NotImplementedError(f"Taxonomy {taxonomy} unsupported")
            new_tag = mapping_dict.get(dialogue_act, "%")  # mapping to literature map
            if new_tag == "%":  # mapping without rhetorical tags (see WS97 mapping guidelines for more details)
                new_tag = mapping_dict.get(
                    dialogue_act.split(",")[0].split(";")[0].split("^")[0].split("(")[0].replace("*", "").replace("@",
                                                                                                                  ""),
                    mapping_dict["%"])
        return new_tag
