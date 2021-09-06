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
MIDAS class: loads the corpus into Utterances. Provides methods
to dump the corpus in CSV format with different annotations
"""


class MIDAS(Corpus):
    def __init__(self, midas_folder, taxonomy: Taxonomy):
        Corpus.__init__(self, "MIDAS", midas_folder, taxonomy)
        self.files = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
        corpus = self.load_corpus(midas_folder)
        self.utterances = self.parse_corpus(corpus)

    def validate_corpus(self, folder):
        return (
            os.path.exists(folder)
            and os.path.exists(f"{folder}/train.txt")
            and os.path.exists(f"{folder}/test.txt")
            and os.path.exists(f"{folder}/dev.txt")
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
                "Check https://catalog.ldc.upenn.edu/ldc97s62 "
                "for info on how to obtain the complete SWDA corpus."
            )
            return
        corpus = {}
        for key in self.files:
            with open(f"{folder}/{self.files[key]}") as f:
                corpus[key] = f.readlines()
        return corpus

    def parse_corpus(self, conversations):
        """
        parses a list of conversations into a list of Utterances.
        Conversations use the following format:
        chatbot utterance : previous user utterance > current user utterance ## DA1;DA2
        :return:
        """
        utterances = {"train": [], "test": [], "dev": []}
        for destination in self.files:
            for row in conversations[destination]:
                if "##" not in row:
                    continue  # no DA: skip row
                user_utterance = (
                    row.split("##")[0]
                    .split(":")[1]
                    .replace(" > ", " ")
                    .replace("EMPTY", "")
                    .strip()
                )
                prev = row.split(":")[0]
                tags = [
                    t.strip() for t in row.split("##")[1].split(";") if t.strip() != ""
                ]
                utterances[destination].append(
                    Utterance(
                        text=user_utterance,
                        tags=self.da_to_taxonomy(tags, self.taxonomy, []),
                        context=[
                            Utterance(
                                prev,
                                self.da_to_taxonomy(["Other"], self.taxonomy, []),
                                [],
                                1,
                            )
                        ],
                        speaker_id=0,
                    )
                )
        return utterances

    @staticmethod
    def da_to_taxonomy(
        tags: List[str], taxonomy: Taxonomy, context: List[str]
    ) -> List[Tag]:
        if taxonomy == Taxonomy.MIDAS:
            mapping_dict = {
                "appreciation": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.Appreciation,
                ),
                "open_question_opinion": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.OpinionQuestion,
                ),
                "apology": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Apology,
                ),
                "command": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.TaskCommand,
                ),
                "pos_answer": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.PositiveAnswer,
                ),
                "nonsense": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Nonsense,
                ),
                "closing": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Closing,
                ),
                "opinion": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.GeneralOpinion,
                ),
                "statement": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.StatementNonOpinion,
                ),
                "neg_answer": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.NegativeAnswer,
                ),
                "dev_command": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.InvalidCommand,
                ),
                "open_question_factual": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.FactualQuestion,
                ),
                "back-channeling": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.BackChanneling,
                ),
                "comment": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.Comment,
                ),
                "yes_no_question": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.YesNoQuestion,
                ),
                "thanking": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Thanks,
                ),
                "respond_to_apology": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.ApologyResponse,
                ),
                "other_answers": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.OtherAnswer,
                ),
                "hold": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Hold,
                ),
                "other": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Other,
                ),
                "complaint": MIDASTag(
                    dimension=MIDASDimension.SemanticRequest,
                    comm_function=MIDASSemanticFunction.Complaint,
                ),
                "opening": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Opening,
                ),
                "abandon": MIDASTag(
                    dimension=MIDASDimension.FunctionalRequest,
                    comm_function=MIDASFunctionalFunction.Abandon,
                ),
            }
        elif taxonomy == Taxonomy.ISO:
            mapping_dict = {
                "appreciation": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
                "open_question_opinion": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.SetQ
                ),
                "apology": ISOTag(
                    dimension=ISODimension.SocialObligation,
                    comm_function=ISOSocialFunction.Apology,
                ),
                "command": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Directive
                ),
                "pos_answer": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
                "nonsense": ISOTag(
                    dimension=ISODimension.Unknown,
                    comm_function=ISOTaskFunction.Unknown,
                ),
                "opinion": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
                "statement": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
                "neg_answer": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
                "dev_command": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Directive
                ),
                "open_question_factual": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.SetQ
                ),
                "back-channeling": ISOTag(
                    dimension=ISODimension.Feedback,
                    comm_function=ISOFeedbackFunction.Feedback,
                ),
                "comment": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
                "yes_no_question": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.PropQ
                ),
                "thanking": ISOTag(
                    dimension=ISODimension.SocialObligation,
                    comm_function=ISOSocialFunction.Thanking,
                ),
                "respond_to_apology": ISOTag(
                    dimension=ISODimension.SocialObligation,
                    comm_function=ISOSocialFunction.Apology,
                ),
                "other_answers": ISOTag(
                    dimension=ISODimension.Task, comm_function=ISOTaskFunction.Statement
                ),
            }
        else:
            raise NotImplementedError(f"Taxonomy {taxonomy} unsupported")
        tags = [
            mapping_dict.get(da, mapping_dict["nonsense"]) for da in tags
        ]  # mapping to literature map
        if taxonomy != Taxonomy.MIDAS:
            # taxonomies that aren't MIDAS don't support multiple tags for the same dimension
            tags = list({tag.dimension: tag for tag in tags}.values())
        return tags
