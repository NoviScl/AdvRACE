## AdvRACE: Benchmarking Robustness of Machine Reading Comprehension Models

arXiv link: https://arxiv.org/abs/2004.14004

This repo contains the constructed datasets as well as the source codes for adversarial attack generation and model evaluation. 


### Dependencies

Required dependencies are in the requirements.txt file:
```
pip install requirements.txt
```

### AdvRACE Datasets

There are four adversarial test sets in the AdvRACE benchmark: AddSent, CharSwap, Distractor Extraction (DE) and Distractor Generation (DG). They are in the `final_distractor_datasets` directory. We have also included the original RACE test set (high+middle) in the same directory for easy access. You can evaluate your models on these test sets for robustness comparison.

The adversarial test sets are all stored in json format, named as `test_dis.json` in the respective sub-directory.

In addition, we also provide the other two adversarial test sets that did not pass our human evaluation: papaphrase and wordReplace. Ther are in the `ununsed_datasets` directory, in case you are interested. We do not use them for model evaluation due to the relatively poor quality.


### Adversarial Attack Construction

To facilitate future work, we also release the codes for constructing the adversarial attacks in the `Attacks` directory.

Some of the attack codes are borrowed from existing works, and we'd like to acknowledge them:

AddSent - from Jia&Liang's work: https://github.com/robinjia/adversarial-squad

Paraphrase - from Mohit Iyyer et al.'s work: https://github.com/miyyer/scpn

WordReplace - from Yuan Zang et al.'s work: https://github.com/thunlp/SememePSO-Attack



### Training and Evaluation 

We were using Huggingface's Transformers (2.3.0) and we provide and corresponding training and evaluation scripts that we used.

The training should be done with the original RACE training set, and in our work we used the original RACE dev set for hyper-parameter tuning. You can download the original RACE dataset [here](http://www.cs.cmu.edu/~glai1/data/race/).

To train your model on RACE, run:
```
sh run_training.sh
```

Note that in the script you should specify the directory where you put the training data, as well as the pre-trained model that you want to use.

After you have trained your own model, you can then evaluate on the original RACE test set and AdvRACE test sets for comparison. For model evaluation, run:
```
sh run_testing.sh
```

### Leaderboard 

We don't have a leaderboard for now, but we will consider setting up one in the future when there are more results on AdvRACE. Before that, feel free to submit a pull request to submit your results onto this page so that other people working on it can also keep track of the progress.

(TODO: Add our results, CharBERT results, and maybe PQ-AT's results on AdvRACE here.)

