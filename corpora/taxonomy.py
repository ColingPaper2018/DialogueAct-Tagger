from enum import Enum
from typing import Union
from dataclasses import dataclass


class Layer(Enum):
    Dimension = 0
    CommFunction = 1


class Taxonomy(Enum):
    AMI = "ami",
    ISO = "iso"

    @staticmethod
    def from_str(taxonomy: str) -> "Taxonomy":
        for t in Taxonomy:
            if t.value == taxonomy:
                return t
        raise NotImplementedError(f"Unknown taxonomy: {taxonomy}")


# ISO Standard
class ISODimension(Enum):
    """
    Enum for the Dimensions of the Dialogue Act Taxonomy. Currently supported dimensions are
    Task
    SocialObligation
    Feedback
    """
    Task = 0
    SocialObligation = 1
    Feedback = 2


class ISOSocialFunction(Enum):
    """
    Tags for the Social Obligation dimension
    """
    Thanking = 0
    Salutation = 1
    Apology = 2


class ISOTaskFunction(Enum):
    """
    Tags for the Task Dimension
    """
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
    Feedback = 0


@dataclass
class ISOTag:
    dimension: ISODimension
    comm_function: Union[ISOFeedbackFunction, ISOSocialFunction, ISOTaskFunction]


# AMI Corpus
class AMIFunction:
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


# Tag type for taxonomies
Tag = Union[ISOTag, AMITag]
