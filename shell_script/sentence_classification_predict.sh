export BERT_BASE_DIR=/root/uncased_L-12_H-768_A-12
export GLUE_DIR=/root/glue_data
export TRAINED_CLASSIFIER=/root/bert_output/sentence_class_bert_base_mrpc

name=`date "+%Y%m%d%H%M%S"`
logfile=logs/sentence_classification_predict_${name}.log
echo ${logfile}

nohup python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/root/bert_output/sentence_class_bert_base_mrpc 1>${logfile} 2>&1 &
