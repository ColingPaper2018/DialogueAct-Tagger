#!/bin/bash

#1) Install python dependencies

pip install -r requirements.txt
python -m spacy download en

#2) Download publicly available corpora
if [ ! -d "data/Maptask/" ] ; then
    mkdir data/Maptask
fi

if [ ! -d "data/AMI/" ] ; then
    mkdir data/AMI
fi

if [ ! -d "data/DailyDialog" ] ; then
    mkdir data/DailyDialog
fi

if [ ! -d "data/Maptask/maptaskv2-1" ] ; then
	echo "Downloading the Maptask corpus"
	wget http://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip && unzip -q hcrcmaptask.nxtformatv2-1.zip && rm -rf hcrcmaptask.nxtformatv2-1.zip && mv maptaskv2-1 data/Maptask/maptaskv2-1
fi

if [ ! -d "data/AMI/corpus" ] ; then
	echo "Downloading the AMI corpus"
	 mkdir corpus && cd corpus && wget http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip && unzip -q ami_public_manual_1.6.2.zip && rm -f ami_public_manual_1.6.2.zip && cd .. && mv corpus data/AMI/corpus
fi


if [ ! -d "data/DailyDialog/corpus" ] ; then
	echo "Downloading the DailyDialog corpus"
         mkdir corpus && cd corpus && wget http://yanran.li/files/ijcnlp_dailydialog.zip && unzip -q ijcnlp_dailydialog.zip && cd ijcnlp_dailydialog && unzip train.zip && unzip test.zip && unzip validation.zip && rm -f train.zip test.zip validation.zip && cd .. && rm -f ijcnlp_dailydialog.zip && cd .. && mv corpus/ijcnlp_dailydialog data/DailyDialog/corpus
fi