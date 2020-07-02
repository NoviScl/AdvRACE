# Change to your own model checkpoint path!
MODEL_PATH=/ps2/intern/clsi/output_race_roberta_advTrain_charSwap100/checkpoint-52000

~/anaconda3/bin/python transformers/examples/run_multiple_choice.py \
  --model_type roberta \
  --task_name new-race \
  --model_name_or_path ${MODEL_PATH} \
  --do_test \
  --data_dir final_distractor_datasets/orig/ \
  --logging_steps 500 \
  --save_steps 1000 \
  --max_seq_length 512 \
  --output_dir ./output_race_evals/ \
  --per_gpu_eval_batch_size 2 \
  --overwrite_output_dir \
  --overwrite_cache \
  --fp16

~/anaconda3/bin/python transformers/examples/run_multiple_choice.py \
  --model_type roberta \
  --task_name new-race \
  --model_name_or_path ${MODEL_PATH} \
  --do_test \
  --data_dir final_distractor_datasets/AddSent/ \
  --logging_steps 500 \
  --save_steps 1000 \
  --max_seq_length 512 \
  --output_dir ./output_race_evals/ \
  --per_gpu_eval_batch_size 2 \
  --overwrite_output_dir \
  --overwrite_cache \
  --fp16

~/anaconda3/bin/python transformers/examples/run_multiple_choice.py \
  --model_type roberta \
  --task_name new-race \
  --model_name_or_path ${MODEL_PATH} \
  --do_test \
  --data_dir final_distractor_datasets/charSwap/ \
  --logging_steps 500 \
  --save_steps 1000 \
  --max_seq_length 512 \
  --output_dir ./output_race_evals/ \
  --per_gpu_eval_batch_size 2 \
  --overwrite_output_dir \
  --overwrite_cache \
  --fp16

~/anaconda3/bin/python transformers/examples/run_multiple_choice.py \
  --model_type roberta \
  --task_name new-race \
  --model_name_or_path ${MODEL_PATH} \
  --do_test \
  --data_dir final_distractor_datasets/DE/ \
  --logging_steps 500 \
  --save_steps 1000 \
  --max_seq_length 512 \
  --output_dir ./output_race_evals/ \
  --per_gpu_eval_batch_size 2 \
  --overwrite_output_dir \
  --overwrite_cache \
  --fp16

~/anaconda3/bin/python transformers/examples/run_multiple_choice.py \
--model_type roberta \
--task_name new-race \
--model_name_or_path ${MODEL_PATH} \
--do_test \
--data_dir final_distractor_datasets/DG/ \
--logging_steps 500 \
--save_steps 1000 \
--max_seq_length 512 \
--output_dir ./output_race_evals/ \
--per_gpu_eval_batch_size 2 \
--overwrite_output_dir \
--overwrite_cache \
--fp16
