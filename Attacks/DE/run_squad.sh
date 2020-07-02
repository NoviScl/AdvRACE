# export SQUAD_DIR=spanRACE

# export PATH="~/anaconda3/bin:\$PATH"

#training, do eval on dev set first; use the trained model to predict nbest on test set later.
~/anaconda3/bin/python run_squad.py \
  --model_type albert \
  --model_name_or_path /ps2/intern/clsi/ALBERT/albert-xxlarge-pytorch_model.bin \
  --config_name /ps2/intern/clsi/ALBERT/albert-xxlarge-config.json \
  --tokenizer_name /ps2/intern/clsi/ALBERT/albert-xxlarge-spiece.model \
  --train_file /ps2/intern/clsi/spanRACE/train_spanRace.json \
  --predict_file /ps2/intern/clsi/spanRACE/dev_spanRace.json \
  --do_train \
  --do_eval \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3.0 \
  --learning_rate 2e-5 \
  --warmup_steps 1000 \
  --logging_steps 1000 \
  --save_steps 1000 \
  --max_seq_length 512 \
  --output_dir ./output_albert_spanRACE_newV2/ \
  --cache_dir ./cache_albert_spanRACE_newV2/ \
  --overwrite_output_dir \
  --n_best_size 2 \
  --fp16

# ~/anaconda3/bin/python run_squad.py \
#   --model_type roberta \
#   --model_name_or_path /ps2/intern/clsi/RoBERTa \
#   --train_file /ps2/intern/clsi/spanRACE/train_spanRace.json \
#   --predict_file /ps2/intern/clsi/spanRACE/dev_spanRace.json \
#   --do_train \
#   --do_eval \
#   --per_gpu_train_batch_size 2 \
#   --per_gpu_eval_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --num_train_epochs 3.0 \
#   --learning_rate 1.5e-5 \
#   --adam_epsilon 1e-6 \
#   --weight_decay 0.1 \
#   --warmup_steps 1000 \
#   --logging_steps 1000 \
#   --save_steps 1000 \
#   --eval_all_checkpoints \
#   --max_seq_length 512 \
#   --n_best_size 2 \
#   --output_dir ./output_roberta_spanRACE/ \
#   --cache_dir ./cache_roberta_spanRACE/ \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --seed 12 \
#   --fp16

# ~/anaconda3/bin/python transformers/examples/run_multiple_choice.py \
#   --model_type roberta \
#   --task_name new-race \
#   --model_name_or_path /ps2/intern/clsi/RoBERTa \
#   --do_train \
#   --do_test \
#   --data_dir /ps2/intern/clsi/extractive/adv_training/ \
#   --learning_rate 1e-5 \
#   --adam_epsilon 1e-6 \
#   --weight_decay 0.1 \
#   --warmup_steps 1000 \
#   --num_train_epochs 3.0 \
#   --logging_steps 1000 \
#   --save_steps 1000 \
#   --eval_all_checkpoints \
#   --max_seq_length 512 \
#   --output_dir ./output_race_roberta_advTrain_extractive25_seed20/ \
#   --per_gpu_eval_batch_size 2 \
#   --per_gpu_train_batch_size 2 \
#   --gradient_accumulation_steps 8 \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --seed 20 \
#   --fp16



## increase nbest size when doing inference on test set!


## generate extractive train file, keep nbest=20
# ~/anaconda3/bin/python run_squad.py \
#   --model_type albert \
#   --model_name_or_path /ps2/intern/clsi/output_albert_spanRACE_newV2/checkpoint-12000 \
#   --config_name /ps2/intern/clsi/ALBERT/albert-xxlarge-config.json \
#   --tokenizer_name /ps2/intern/clsi/ALBERT/albert-xxlarge-spiece.model \
#   --predict_file /ps2/intern/clsi/spanRACE_orig/train_spanRace.json \
#   --do_eval \
#   --per_gpu_eval_batch_size 1 \
#   --logging_steps 1000 \
#   --save_steps 1000 \
#   --max_seq_length 512 \
#   --output_dir ./output_albert_spanRACE_newV2_train_nbest/ \
#   --cache_dir ./cache_albert_spanRACE_newV2/ \
#   --n_best_size 20 \
#   --fp16



# ~/anaconda3/bin/python run_squad.py \
#   --model_type albert \
#   --model_name_or_path /ps2/intern/clsi/output_albert_spanRACE_newV2/checkpoint-12000 \
#   --config_name /ps2/intern/clsi/ALBERT/albert-xxlarge-config.json \
#   --tokenizer_name /ps2/intern/clsi/ALBERT/albert-xxlarge-spiece.model \
#   --predict_file /ps2/intern/clsi/spanRACE_orig/dev_spanRace.json \
#   --do_eval \
#   --per_gpu_eval_batch_size 1 \
#   --logging_steps 1000 \
#   --save_steps 1000 \
#   --max_seq_length 512 \
#   --output_dir ./output_albert_spanRACE_newV2_dev_nbest/ \
#   --cache_dir ./cache_albert_spanRACE_newV2/ \
#   --n_best_size 20 \
#   --fp16



# ~/anaconda3/bin/python run_squad.py \
#   --model_type albert \
#   --model_name_or_path /ps2/intern/clsi/albert-large-checkpoint-0 \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --train_file /ps2/intern/clsi/spanRACE/train_spanRace.json \
#   --predict_file /ps2/intern/clsi/spanRACE/dev_spanRace.json \
#   --config_name /ps2/intern/clsi/albert-xxlarge-v2-config.json \
#   --tokenizer_name /ps2/intern/clsi/albert-xxlarge-v2-spiece.model \
#   --per_gpu_train_batch_size 1 \
#   --gradient_accumulation_steps 12 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 3.0 \
#   --max_seq_length 512 \
#   --doc_stride 128 \
#   --output_dir ./output_spanRACE_ALBERT/ \
#   --overwrite_cache \
#   --overwrite_output_dir