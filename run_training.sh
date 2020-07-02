MODEL_PATH=/ps2/intern/clsi/RoBERTa-Large # change to your own model name/path
DATA_PATH=/ps2/intern/clsi/RACE # change to your own data path!

# training with RoBERTa
~/anaconda3/bin/python transformers/examples/run_multiple_choice.py \
  --model_type roberta \
  --task_name race \
  --model_name_or_path ${MODEL_PATH} \
  --do_train \
  --do_eval \
  --data_dir ${DATA_PATH} \
  --learning_rate 1e-5 \
  --adam_epsilon 1e-6 \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --num_train_epochs 5.0 \
  --logging_steps 1000 \
  --save_steps 2000 \
  --eval_all_checkpoints \
  --max_seq_length 512 \
  --output_dir ./output_race_roberta/ \
  --per_gpu_eval_batch_size 4 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed 12 \
  --fp16