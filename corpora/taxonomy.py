from enum import Enum
from typing import Union
from dataclasses import dataclass


# ISO Standard
class ISODimension(Enum):
    """
    Enum for the Dimensions of the Dialogue Act Taxonomy. Currently supported dimensions are
    Task
    SocialObligation
    Feedback
    """
    Unknown = -1
    Task = 0
    SocialObligation = 1
    Feedback = 2


class ISOSocialFunction(Enum):
    """
    Tags for the Social Obligation dimension
    """
    Unknown = -1
    Thanking = 0
    Salutation = 1
    Apology = 2


class ISOTaskFunction(Enum):
    """
    Tags for the Task Dimension
    """
    Unknown = -1
    Statement = 0
    PropQ = 1
    SetQ = 2
    ChoiceQ = 3
    Directive = 4
    Commissive = 5


class ISOFeedbackFunction(Enum):
    """
    Tags for the Feedback dimension
    """
    Unknown = -1
    Feedback = 0


@dataclass
class ISOTag:
    dimension: ISODimension
    comm_function: Union[ISOFeedbackFunction, ISOSocialFunction, ISOTaskFunction]


# AMI Corpus
class AMIFunction(Enum):
    Unknown = -1
    Backchannel = 0
    Stall = 1
    Fragment = 2
    Inform = 3
    Suggest = 4
    Assess = 5
    ElicitInform = 6
    ElicitOfferOrSuggest = 7
    ElicitAssess = 8
    ElicitCommentAboutUnderstanding = 9
    CommentAboutUnderstanding = 10
    Offer = 11
    BePositive = 12
    BeNegative = 13
    Other = 14


@dataclass
class AMITag:
    comm_function: AMIFunction


# Maptask Corpus
class MaptaskFunction(Enum):
    Unknown = -1
    Instruct = 0
    Explain = 1
    Check = 2
    Align = 3
    QueryYN = 4
    QueryW = 5
    Acknowledge = 6
    ReplyY = 7
    ReplyN = 8
    ReplyW = 9
    Clarify = 10


@dataclass
class MaptaskTag:
    comm_function: MaptaskFunction


# SWDA Corpus
class SWDAFunction(Enum):
    Uninterpretable = -1
    StatementNonOpinion = 0
    Acknowledge = 1
    StatementOpinion = 2
    AgreeAccept = 3
    Abandoned = 4
    Appreciation = 5
    YesNoQuestion = 6
    NonVerbal = 7
    YesAnswers = 8
    ConventionalClosing = 9
    Thanking = 10
    WhQuestion = 11
    NoAnswers = 12
    ResponseAcknowledgement = 13
    Hedge = 14
    DeclarativeYNQuestion = 15
    Other = 16
    BackchannelInQuestionForm = 17
    Quotation = 18
    SummarizeReformulate = 19
    AffirmativeNonYesAnswers = 20
    ActionDirective = 21
    CollaborativeCompletion = 22
    RepeatPhrase = 23
    OpenQuestion = 24
    RhetoricalQuestions = 25
    Hold = 26
    Reject = 27
    NegativeNonNoAnswers = 28
    SignalNonUnderstanding = 29
    OtherAnswers = 30
    ConventionalOpening = 31
    OrClause = 32
    DispreferredAnswers = 33
    ThirdPartyTalk = 34
    OffersOptionsCommits = 35
    SelfTalk = 36
    Downplayer = 37
    MaybeAcceptPart = 38
    TagQuestion = 39
    DeclarativeWhQuestion = 40
    Apology = 41


@dataclass
class SWDATag:
    comm_function: SWDAFunction


# Tag type for taxonomies
Tag = Union[ISOTag, AMITag, MaptaskTag, SWDATag]


class Taxonomy(Enum):
    AMI = AMITag
    ISO = ISOTag
    Maptask = MaptaskTag
    SWDA = SWDATag

    @staticmethod
    def from_str(taxonomy: str) -> "Taxonomy":
        if taxonomy.lower() == "amitag":
            return Taxonomy.AMI
        elif taxonomy.lower() == "isotag":
            return Taxonomy.ISO
        elif taxonomy.lower() == "maptasktag":
            return Taxonomy.Maptask
        elif taxonomy.lower() == "swdatag":
            return Taxonomy.SWDA
        else:
            raise NotImplementedError(f"Unknown taxonomy: {taxonomy}")

    def to_str(self):
        if self.value == AMITag:
            return "AMITag"
        elif self.value == ISOTag:
            return "ISOTag"
        elif self.value == MaptaskTag:
            return "MaptaskTag"
        elif self.value == SWDATag:
            return "SWDATag"
        else:
            raise ValueError("Taxonomy value changed to an unexpected value")
