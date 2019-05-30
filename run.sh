#MODEL_PATH=data/pretrained_model/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt
MODEL_PATH=/local-scratch/pooya/attention_explanation/models/iwslt_de_en/model_step_10000.pt
TEST_SRC=data/fairseq_de_en/iwslt14.tokenized.de-en/test.de
#TEST_TRG=data/fairseq_de_en/iwslt14.tokenized.de-en/test.en

#TEST_SRC=test.de
#TEST_TRG=test.en

python OpenNMT-py/translate.py -model $MODEL_PATH -src $TEST_SRC -beam_size 1 -gpu 0 #-tgt $TEST_TRG
