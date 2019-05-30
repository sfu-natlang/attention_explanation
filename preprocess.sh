#!/bin/bash

TRAIN_SRC=data/fairseq_de_en/iwslt14.tokenized.de-en/train.de
TRAIN_TGT=data/fairseq_de_en/iwslt14.tokenized.de-en/train.en
VALID_SRC=data/fairseq_de_en/iwslt14.tokenized.de-en/valid.de
VALID_TGT=data/fairseq_de_en/iwslt14.tokenized.de-en/valid.en

SAVE_DATA=models/iwslt-de-en

python OpenNMT-py/preprocess.py -train_src $TRAIN_SRC -train_tgt $TRAIN_TGT -valid_src $VALID_SRC -valid_tgt $VALID_TGT -save_data $SAVE_DATA
