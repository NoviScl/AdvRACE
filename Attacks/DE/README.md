To generate the DE attack (i.e., extract good distractors from the passage):

1) Convert RACE into spanRACE format. (convertRACE.py)
2) Train a QA model in Squad style (run_squad.sh), note that you need to change the cache directory accordingly.
3) After getting the trained model (I selected checkpoint based on QA performance on dev set). You can run the inference script on the test set (which should NOT have the correct answer inserted into the passage). The script will store the top 20 candidates. You can the select 3 valid ones (I was using a rather simple filtering prodecure) to get 3 top distractors (use the convert scritpt).
4) transform_distractor.py can be used for distractor post-processing.