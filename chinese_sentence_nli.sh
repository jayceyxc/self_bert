export BERT_BASE_DIR=/root/chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
export XNLI_DIR=/root/xnli_data

name=`date "+%Y%m%d%H%M%S"`
logfile=logs/chinese_sentence_nli_${name}.log
echo ${logfile}

nohup python run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/root/bert_output/chinese_nli_bert_base_xnli 1>${logfile} 2>&1 &
