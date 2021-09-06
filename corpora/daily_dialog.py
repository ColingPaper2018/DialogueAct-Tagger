import os
import re
from corpora.corpus import Corpus
import logging
from corpora.taxonomy import (
    Tag,
    Taxonomy,
    ISOTag,
    ISODimension,
    ISOTaskFunction,
    ISOFeedbackFunction,
    ISOSocialFunction,
    MIDASTag,
    MIDASDimension,
    MIDASSemanticFunction,
    MIDASFunctionalFunction,
)
from typing import List
from corpora.corpus import Utterance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ISO_DA")


"""
DailyDialog class: loads the corpus into Utterances. Provides methods
to dump the corpus in CSV format with different annotations
"""


class DailyDialog(Corpus):
    def __init__(self, daily_dialog_folder, taxonomy: Taxonomy):
        Corpus.__init__(self, "DailyDialog", daily_dialog_folder, taxonomy)
        corpus = self.load_corpus(daily_dialog_folder)
        self.utterances = self.parse_corpus(corpus)

    def validate_corpus(self, folder):
        for split in ["train", "test", "validation"]:
            return (
                os.path.exists(folder)
                and os.path.exists(f"{folder}/corpus/{split}/dialogues_act_{split}.txt")
                and os.path.exists(f"{folder}/corpus/{split}/dialogues_{split}.txt")
            )

    def load_corpus(self, folder):
        # check whether the folder contains a valid Switchboard installation
        try:
            assert self.validate_corpus(folder)
        except AssertionError:
            logging.warning(
                f"The folder {folder} does not contain some important files."
            )
            logging.info(
                "Check http://yanran.li/dailydialog.html "
                "for info on how to obtain the complete DailyDialog corpus."
            )
            return
        corpus = {"train": [], "test": [], "validation": []}
        for key in corpus:
            with open(f"{folder}/corpus/{key}/dialogues_act_{key}.txt") as acts_f:
                acts = [line.strip() for line in acts_f.readlines()]
            with open(f"{folder}/corpus/{key}/dialogues_{key}.txt") as texts_f:
                texts = [line.strip() for line in texts_f.readlines()]
            for conv, tags in zip(texts, acts):
                conversation = []
                utterances = [u.strip() for u in conv.split("__eou__")]
                tags = [t.strip() for t in tags.split(" ")]
                for idx, (utt, tag) in enumerate(zip(utterances, tags)):
                    conversation.append([utt, tag, idx % 2])
                corpus[key].append(conversation)
        return corpus

    def parse_corpus(self, corpus):
        """
        parses a list of conversations into a list of Utterances.
        Conversations use the following format:
        chatbot utterance : previous user utterance > current user utterance ## DA1;DA2
        :return:
        """
        utterances = {"train": [], "test": [], "validation": []}
        for key in corpus:
            for conversation in corpus[key]:
                for idx, utterance in enumerate(conversation):
                    utterances[key].append(
                        Utterance(
                            text=utterance[0],
                            tags=self.da_to_taxonomy([utterance[1]], self.taxonomy, []),
                            context=[
                                Utterance(
                                    conversation[idx - 1][0],
                                    self.da_to_taxonomy(
                                        [conversation[idx - 1][1]], self.taxonomy, []
                                    ),
                                    [],
                                    conversation[idx - 1][2],
                                )
                            ]
                            if idx > 0
                            else [
                                Utterance(
                                    text="",
                                    tags=self.da_to_taxonomy(["0"], self.taxonomy, []),
                                    context=[],
                                    speaker_id=0,
                                )
                            ],
                            speaker_id=utterance[2],
                        )
                    )
        return utterances

    @staticmethod
    def da_to_taxonomy(
        tags: List[str], taxonomy: Taxonomy, context: List[str]
    ) -> List[Tag]:
        if taxonomy == Taxonomy.ISO:
            mapping_dict = {
                "0": ISOTag(
                    dimension=ISODimension.Unknown,
                    comm_function=ISOTaskFunction.Unknown,
                ),
                "1": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
                "3": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Directive
                ),
                "4": ISOTag(
                    dimension=ISODimension.Task,
                    comm_function=ISOTaskFunction.Commissive,
                ),
            }
        else:
            raise NotImplementedError(f"Taxonomy {taxonomy} unsupported")
        tags = [
            mapping_dict.get(da, mapping_dict["0"]) for da in tags
        ]  # mapping to literature map
        return tags
