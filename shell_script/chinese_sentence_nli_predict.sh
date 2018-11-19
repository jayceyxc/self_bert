#!/usr/bin/env bash
export BERT_BASE_DIR=/root/chinese_L-12_H-768_A-12
export XNLI_DIR=/root/xnli_data
export TRAINED_CLASSIFIER=/root/bert_output/chinese_nli_bert_base_xnli

name=`date "+%Y%m%d%H%M%S"`
logfile=logs/chinese_sentence_nli_predict_${name}.log
echo ${logfile}

nohup python run_classifier.py \
  --task_name=XNLI \
  --do_predict=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/root/bert_output/chinese_nli_bert_base_xnli 1>${logfile} 2>&1 &
