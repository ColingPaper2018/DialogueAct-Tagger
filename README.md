# ISO-Standard-DA
A resource to create an ISO-compliant Dialog Act Tagger using publicly available data

# Getting started

This repository contains a set of utilities which can be used to train and test a fully functional Dialogue Act Tagger compliant to the new ISO DA scheme. 
This software only uses publicly available datasets to train the models. However, some of these resources may require authorizations before they are used. Please check that you have all the available data before using this code. You can find information on how to obtain the required corpora on their official websites:

1. For Switchboard: https://catalog.ldc.upenn.edu/docs/LDC97S62/
2. For Oasis BT: http://groups.inf.ed.ac.uk/oasis/
3. For Maptask http://groups.inf.ed.ac.uk/maptask/
4. For VerbMobil2: https://www.phonetik.uni-muenchen.de/Bas/BasVM2eng.html
5. For AMI: http://groups.inf.ed.ac.uk/ami/corpus/

Each corpus folder has a `data` subfolder which must contain the appropriate resource's unzipped parent folder.
The file `DialogueActTrain.py` contains a small main method which can be used to train a Dialogue Act Tagger. 
If you do not have access to some of the resources you can easily comment the corresponding line to avoid using that resource when training. 
Your DA tagger will then be stored in the `models` folder. You can then use the `DialogActTagger` class in the self-named Python file to test it and use it.
The code was written using Python3 and requires spaCy 1.8 and the latest version of Scikit-learn to run.

# The Corpus class and the corpora mappers

The main component of the code architecture is the `Corpus` interface and their extensions, which handle conversion of the corporas' original scheme to the ISO standard. The interface is built around four main steps:

* Loading (usually handled in the constructor)
* Converting to CSV (`load_csv` method). This is necessary since some of these resources were annotated in a quite unconvenient XML format which makes it hard to read for a human annotator.
* Mapping to ISO standard (`create_csv` method)
* Dumping (a separate method was implemented for each ISO dimension, plus one to just output the original corpus annotation in CSV).

The csv is comma-separated and has a very simple structure. Each row is a tuple in the form

`(sentence, DA tag, previous DA tag,  segment, additional info, previous additional info)`

where

* sentence is the annotated utterance
* DA tag is the corresponding DA tag
* previous DA tag is the DA tag of the previous sentence
* segment is an index which encodes the logical segment to which the utterance belongs (many of these corpora contain multi-utterance DA annotations).
* additional info contains a JSON with additional information required to map this corpus (for example, for Oasis it contains an additional label used to encode DAs)
* previous additional info contains the additional info for the previous sentence

The Corpus interface can be followed to implement mapping to additional dialogue corpora, which then will be usable in the `DialogueActTrain` class, which handles training of the statistical model.

# Training a model

Training a model is very easy, and is handled by the `DialogueActTrain` class. The class has a `train_all` method which trains a complete model for ISO DA annotation and dumps it in the `output_folder` folder. Features for the single models can be enabled/disabled at will, and new features are easy to add since the code uses extendible Scikit Learn Pipelines. 
Training is done using Support Vector Machine (LinearSVC) for each classifier of the pipeline. Dumping is done by using the `pickle` library, which is included in every version of Python > 2.6

# Using the DA tagger

Once the training is complete, the DA tagger is ready to use. The `DialogueActTagger` class loads a trained model and exposes methods to tag a sentence with the complete taxonomy or to just tag a single dimension. The confidence of each model is also returned by these methods to give a feedback on the reliability of the prediction. 
