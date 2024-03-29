from enum import Enum
from typing import Union
from dataclasses import dataclass


class Tagset:
    @staticmethod
    def get_dimension_taxonomy() -> Enum:
        raise NotImplementedError()

    @staticmethod
    def get_comm_taxonomy_given_dimension(dimension_value: int) -> Enum:
        raise NotImplementedError()


# ISO Standard
class ISODimension(Enum):
    """
    Enum for the Dimensions of the Dialogue Act Taxonomy. Currently supported dimensions are
    Task
    SocialObligation
    Feedback
    """

    Unknown = 0
    Task = 1
    SocialObligation = 2
    Feedback = 3

    @staticmethod
    def values():
        return {0: "Unknown", 1: "Task", 2: "SocialObligation", 3: "Feedback"}


class ISOSocialFunction(Enum):
    """
    Tags for the Social Obligation dimension
    """

    Unknown = 0
    Thanking = 1
    Salutation = 2
    Apology = 3

    @staticmethod
    def values():
        return {0: "Unknown", 1: "Thanking", 2: "Salutation", 3: "Apology"}


class ISOTaskFunction(Enum):
    """
    Tags for the Task Dimension
    """

    Unknown = 0
    Statement = 1
    PropQ = 2
    SetQ = 3
    ChoiceQ = 4
    Directive = 5
    Commissive = 6

    @staticmethod
    def values():
        return {
            0: "Unknown",
            1: "Statement",
            2: "PropQ",
            3: "SetQ",
            4: "ChoiceQ",
            5: "Directive",
            6: "Commissive",
        }


class ISOFeedbackFunction(Enum):
    """
    Tags for the Feedback dimension
    """

    Unknown = 0
    Feedback = 1

    @staticmethod
    def values():
        return {0: "Unknown", 1: "Feedback"}


ISOCommFunction = Union[ISOFeedbackFunction, ISOSocialFunction, ISOTaskFunction]


@dataclass
class ISOTag(Tagset):
    dimension: ISODimension
    comm_function: ISOCommFunction

    def __init__(self, dimension: ISODimension, comm_function: ISOCommFunction):
        self.dimension = dimension
        self.comm_function = comm_function

    @staticmethod
    def get_comm_taxonomy_given_dimension(dimension_value: int):
        dimension_dict = {
            ISODimension.Unknown.value: None,
            ISODimension.Task.value: ISOTaskFunction,
            ISODimension.SocialObligation.value: ISOSocialFunction,
            ISODimension.Feedback.value: ISOFeedbackFunction,
        }
        return dimension_dict[dimension_value]

    @staticmethod
    def get_dimension_taxonomy():
        return ISODimension


# MIDAS
class MIDASDimension(Enum):
    """
    Enum for the Dimensions of the Dialogue Act Taxonomy. Currently supported dimensions are
    Task
    SocialObligation
    Feedback
    """

    Unknown = 0
    FunctionalRequest = 1
    SemanticRequest = 2


class MIDASSemanticFunction(Enum):
    """
    Tags for the Social Obligation dimension
    """

    Unknown = 0
    FactualQuestion = 1
    OpinionQuestion = 2
    YesNoQuestion = 3
    TaskCommand = 4
    InvalidCommand = 5
    Appreciation = 6
    GeneralOpinion = 7
    Complaint = 8
    Comment = 9
    StatementNonOpinion = 10
    OtherAnswer = 11
    PositiveAnswer = 12
    NegativeAnswer = 13


class MIDASFunctionalFunction(Enum):
    """
    Tags for the Task Dimension
    """

    Unknown = 0
    Abandon = 1
    Nonsense = 2
    Opening = 3
    Closing = 4
    Hold = 5
    Thanks = 6
    BackChanneling = 7
    Apology = 8
    ApologyResponse = 9
    Other = 10


MIDASCommFunction = Union[MIDASFunctionalFunction, MIDASSemanticFunction]


@dataclass
class MIDASTag(Tagset):
    dimension: MIDASDimension
    comm_function: MIDASCommFunction

    def __init__(self, dimension: MIDASDimension, comm_function: MIDASCommFunction):
        self.dimension = dimension
        self.comm_function = comm_function

    @staticmethod
    def get_comm_taxonomy_given_dimension(dimension_value: int):
        dimension_dict = {
            MIDASDimension.Unknown.value: None,
            MIDASDimension.FunctionalRequest.value: MIDASFunctionalFunction,
            MIDASDimension.SemanticRequest.value: MIDASSemanticFunction,
        }
        return dimension_dict[dimension_value]

    @staticmethod
    def get_dimension_taxonomy():
        return MIDASDimension


# AMI Corpus
class AMIFunction(Enum):
    Unknown = 0
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
    Backchannel = 15


@dataclass
class AMITag(Tagset):
    def __init__(self, comm_function: AMIFunction):
        self.dimension = None
        self.comm_function: comm_function

    @staticmethod
    def get_comm_taxonomy_given_dimension(dimension_value: int = 0):
        return AMIFunction

    @staticmethod
    def get_dimension_taxonomy():
        return None


# Maptask Corpus
class MaptaskFunction(Enum):
    Unknown = 0
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
    Instruct = 11


@dataclass
class MaptaskTag(Tagset):
    def __init__(self, comm_function: MaptaskFunction):
        self.dimension = None
        self.comm_function = comm_function

    @staticmethod
    def get_comm_taxonomy_given_dimension(dimension_value: int = 0):
        return MaptaskFunction

    @staticmethod
    def get_dimension_taxonomy():
        return None


# Maptask Corpus
class DailyDialogFunction(Enum):
    Unknown = 0
    Inform = 1
    Question = 2
    Directive = 3
    Commissive = 4


@dataclass
class DailyDialogTag(Tagset):
    def __init__(self, comm_function: DailyDialogFunction):
        self.dimension = None
        self.comm_function = comm_function

    @staticmethod
    def get_comm_taxonomy_given_dimension(dimension_value: int = 0):
        return DailyDialogFunction

    @staticmethod
    def get_dimension_taxonomy():
        return None


# SWDA Corpus
class SWDAFunction(Enum):
    Uninterpretable = 0
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
    StatementNonOpinion = 42


@dataclass
class SWDATag(Tagset):
    def __init__(self, comm_function: SWDAFunction):
        self.dimension = None
        self.comm_function = comm_function

    @staticmethod
    def get_comm_taxonomy_given_dimension(dimension_value: int = 0):
        return SWDAFunction

    @staticmethod
    def get_dimension_taxonomy():
        return None


# Tag type for taxonomies
Tag = Union[ISOTag, AMITag, MaptaskTag, SWDATag, DailyDialogTag, MIDASTag]


class Taxonomy(Enum):
    AMI = AMITag
    ISO = ISOTag
    Maptask = MaptaskTag
    SWDA = SWDATag
    MIDAS = MIDASTag
    DailyDialog = MIDASTag

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
        elif taxonomy.lower() == "midastag":
            return Taxonomy.MIDAS
        elif taxonomy.lower() == "dailydialogtag":
            return Taxonomy.DailyDialog
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
        elif self.value == MIDASTag:
            return "MIDASTag"
        elif self.value == DailyDialogTag:
            return "DailyDialogTag"
        else:
            raise ValueError("Taxonomy value changed to an unexpected value")
