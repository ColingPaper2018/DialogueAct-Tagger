#!/bin/bash

#1) Install python dependencies

pip install nltk
pip install scikit-learn
pip install spacy
python -m spacy download en

#2) Download publicly available corpora
if [ ! -d "data/Maptask/" ] ; then
    mkdir data/Maptask
fi

if [ ! -d "data/AMI/" ] ; then
    mkdir data/AMI
fi
if [ ! -d "data/Maptask/maptaskv2-1" ] ; then
	echo "Downloading the Maptask corpus"
	wget http://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip && unzip hcrcmaptask.nxtformatv2-1.zip && rm -rf hcrcmaptask.nxtformatv2-1.zip && mv maptaskv2-1 data/Maptask/maptaskv2-1
fi

if [ ! -d "data/AMI/corpus" ] ; then
	echo "Downloading the AMI corpus"
	 mkdir corpus && cd corpus && wget http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip && unzip ami_public_manual_1.6.2.zip && cd .. && mv corpus data/AMI/corpus
fi
