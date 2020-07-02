DATA_DIR=/ps2/intern/clsi/RACE_seq2seq/
MODEL_RECOVER_PATH=/ps2/intern/clsi/unilm/cleaned_finetuned/bert_save/model.10.bin
#original unilm test set was generated using checkpoint-10, I re-ran with checkpoint-7 to check overfit.
EVAL_SPLIT=test
# export CUDA_VISIBLE_DEVICES=3
export PYTORCH_PRETRAINED_BERT_CACHE=/ps2/intern/clsi/bert_weights/cassed_L-24_H-1024_A-16
# run decoding
~/anaconda3/bin/python src/biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --input_file /ps2/intern/clsi/RACE/train_25.json --split ${EVAL_SPLIT} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_tgt_length 20 --min_len 1 --forbid_duplicate_ngrams --ngram_size 1 \
  --config_path '/ps2/intern/clsi/BERT/bert_weights/cased_L-24_H-1024_A-16/bert_config.json' \
  --batch_size 1 --beam_size 50 --length_penalty 0 --output_file /ps2/intern/clsi/unilm/train_25_dis.json
# run evaluation using our tokenized data as reference
#python qg/eval_on_unilm_tokenized_ref.py --out_file qg/output/qg.test.output.txt
# run evaluation using tokenized data of Du et al. (2017) as reference
#python qg/eval.py --out_file qg/output/qg.test.output.txt

# --input_file /ps2/intern/clsi/unilm/cleaned_distractors/race_test.json
# --input_file /ps2/intern/clsi/RACE/test.json
# --output_file /ps2/intern/clsi/unilm/test_dis.json