#!/usr/bin/env bash
export BERT_BASE_DIR=/root/uncased_L-12_H-768_A-12

# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.

name=`date "+%Y%m%d%H%M%S"`
logfile=logs/run_extract_features_${name}.log
echo ${logfile}

cd ..
nohup python extract_features.py \
  --input_file=data/extract_input.txt \
  --output_file=/root/bert_output/extract_output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8 1>${logfile} 2>&1 &