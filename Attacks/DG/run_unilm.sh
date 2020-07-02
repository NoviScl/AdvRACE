# run fine-tuning
DATA_DIR=/ps2/intern/clsi/unilm/data_seq2seq/
OUTPUT_DIR=/ps2/intern/clsi/unilm/cleaned_finetuned/
MODEL_RECOVER_PATH=/ps2/intern/clsi/unilm-model/unilm1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=/ps2/intern/clsi/bert_weights/cased_L-24_H-1024_A-16
~/anaconda3/bin/python src/biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids \
  --config_path '/ps2/intern/clsi/BERT/bert_weights/cased_L-24_H-1024_A-16/config.json' \
  --data_dir ${DATA_DIR} --src_file src-train.txt --tgt_file tgt-train.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_position_embeddings 512 \
  --mask_prob 0.7 --max_pred 30 \
  --train_batch_size 32 --gradient_accumulation_steps 8 \
  --learning_rate 0.00002 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 10

#export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
#export CUDA_VISIBLE_DEVICES=0,1,2,3
## got bug when using fp16
#
# --topk 10
