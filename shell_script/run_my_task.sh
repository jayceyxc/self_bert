#!/usr/bin/env bash

export BERT_BASE_DIR=/Users/yuxuecheng/bert/base/chinese_L-12_H-768_A-12
export DATA_DIR=/Users/yuxuecheng/bert/data/mytask/

name=`date "+%Y%m%d%H%M%S"`
logfile=logs/my_task_${name}.log
echo ${logfile}

python run_classifier.py \
  --task_name=mytask \
  --do_train=true \
  --do_eval=true \
  --data_dir=${DATA_DIR} \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/Users/yuxuecheng/bert/output/mytask 1>${logfile} 2>&1 &