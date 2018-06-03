#!/bin/bash

#1) Install python dependencies

pip install nltk
pip install scikit-learn
pip install spacy
python -m spacy download en

#2) Download publicly available corpora
if [ ! -d "Maptask/maptaskv2-1" ] ; then
	echo "Downloading the Maptask corpus"
	wget http://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip && unzip hcrcmaptask.nxtformatv2-1.zip && rm -rf hcrcmaptask.nxtformatv2-1.zip && mv maptaskv2-1 Maptask/maptaskv2-1
fi

if [ ! -d "AMI/corpus" ] ; then
	echo "Downloading the AMI corpus"
	 mkdir corpus && cd corpus && wget http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip && unzip ami_public_manual_1.6.2.zip && cd .. && mv corpus AMI/corpus
fi

