# ISO-Standard-DA
A resource to create an ISO-compliant Dialogue Act Tagger using publicly available data.

If you use our code, please cite our COLING 2018 paper ["ISO-Standard Domain-Independent Dialogue Act Tagging for
Conversational Agents"](https://www.aclweb.org/anthology/C18-1300.pdf):

Mezza, S., Cervone, A., Stepanov, E., Tortoreto, G., & Riccardi, G. (2018, August). ISO-Standard Domain-Independent Dialogue Act Tagging for Conversational Agents. In Proceedings of the 27th International Conference on Computational Linguistics (pp. 3539-3551).

# Getting started

This repository contains a set of utilities which can be used to train and test a fully functional Dialogue Act Tagger compliant to the new ISO DA scheme. 
This software only uses publicly available datasets to train the models. However, some of these resources may require authorizations before they are used. Please check that you have all the available data before using this code. You can find information on how to obtain the required corpora on their official websites:

1. For the Switchboard Dialogue Act Corpus: https://web.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz
3. For Maptask http://groups.inf.ed.ac.uk/maptask/
5. For AMI: http://groups.inf.ed.ac.uk/ami/corpus/

_Note: the Oasis BT corpus and the Verbmobil 2 corpus were originally included in the list of supported datasets and are mentioned in our research paper.
However, due to the difficulty of obtaining these corpora and their marginal contribution to the tagger's performances, we
 have decided to stop supporting them in our codebase_.


Each corpus folder has a `data` subfolder which must contain the appropriate resource's unzipped parent folder.
The file `scripts/train.py` contains some basic code which can be used to train a Dialogue Act Tagger.
If you do not have access to some of the resources you can easily comment the corresponding line to avoid using that resource when training. 
Your DA tagger will then be stored in the `models` folder. You can then use one of the `DialogActTagger` class implementations which are located in the `taggers` folder to test a trained model.
The code was written using Python 3.6 and requires spaCy 2 and the latest version of Scikit-learn to run.

# Overview of the core classes

The code handles the import and parsing of the raw corpora, and converts them into high-level classes using either their original taxonomy or the ISO standard taxonomy.

### Taxonomy and Tags

Every supported taxonomy in the codebase is listed in the `Taxonomy` enumeration. A taxonomy has a corresponding tag type (which corresponds to its value in the `Enum`) which implements the `Tag` interface.
Tag is a union of dataclasses whose attributes describe the annotation structure of the corresponding taxonomy. All tags are usually described by a `comm_function`, with some of them (such as the ISO standard) also having a semantic category called a `dimension`

### Utterance and Corpus

The `Utterance` and `Corpus` classes provide abstraction of the raw corpora into high-level classes.
An `Utterance` represents a single datapoint within a dataset. It includes the utterances' `text`, one or more `tags` associated to it, the `speaker_id` for the utterance and a `context` representing contextual information through the history of previous utterances.
A `Corpus` class takes a raw conversational dataset and converts it into a list of `Utterances`, which can then be used to train a dialogue act tagger.
Any implementation of the `Corpus` interface must implement the following methods:

* `validate_corpus`, which ensures all the necessary files are present in the folder.
* `load_corpus`, which loads the raw data from the corpus in a `list` or `dict` object with as little processing as possible.
* `parse_corpus` which converts the raw corpus into a list of `Utterance` objects
* `da_to_taxonomy`, which provides a mapping from the raw corpus tags to a `Tag` of a certain `Taxonomy`.

The Corpus interface can be followed to implement mapping to additional dialogue corpora, which then will be usable in a `Trainer` class, which handles training of the statistical model.

# Training a model

Training a model is very easy, and is handled by the `Trainer` interface. This is a purposely generic interface, which only provides a generic `train` method in order to allow for flexible training approaches using different machine learning methods and libraries. An SVM Trainer is provided, based on Scikit-learn `Pipelines`.

# Using the DA tagger

Once the training is complete, the DA tagger is ready to use. The `DialogueActTagger` class loads a trained model and exposes methods to tag a sentence with the complete taxonomy or to just tag a single dimension. The confidence of each model is also returned by these methods to give a feedback on the reliability of the prediction. 
