#!/bin/bash

DATA_PREFIX=models/iwslt-de-en
MODEL_PATH=/local-scratch/pooya/attention_explanation/models/iwslt_de_en/model
LOG_FILE=logs/iwslt-de-en.log

python OpenNMT-py/train.py -global_attention mlp -data $DATA_PREFIX -save_model $MODEL_PATH -save_checkpoint_steps 5000 -valid_steps 500 -train_steps 50000 -optim adam -learning_rate 0.001 -log_file $LOG_FILE -input_feed 0 -world_size 2 -gpu_ranks 0 1
