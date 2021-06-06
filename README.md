# AdvRACE: Benchmarking Robustness of Machine Reading Comprehension Models

This is the repo containing the processed data and also the code for reproducing the experiments in our paper: Benchmarking Robustness of Machine Reading Comprehension Models. ([arxiv](https://arxiv.org/abs/2004.14004)). ACL 2021 (Findings).


## Dependencies

Required dependencies are in the requirements.txt file:
```
pip install requirements.txt
```

## AdvRACE Datasets

There are four adversarial test sets in the AdvRACE benchmark: AddSent, CharSwap, Distractor Extraction (DE) and Distractor Generation (DG). They are in the `final_distractor_datasets` directory. We have also included the original RACE test set (high+middle) in the same directory for easy access. You can evaluate your models on these test sets for robustness comparison.

The adversarial test sets are all stored in json format, named as `test_dis.json` in the respective sub-directory.

In addition, we also provide the other two adversarial test sets that did not pass our human evaluation: papaphrase and wordReplace. Ther are in the `ununsed_datasets` directory, in case you are interested. We do not use them for model evaluation due to the relatively poor quality.


## Adversarial Attack Construction

To facilitate future work, we also release the codes for constructing the adversarial attacks in the `Attacks` directory.

Some of the attack codes are borrowed from existing works, and we'd like to acknowledge them:

AddSent - from Jia&Liang's work: https://github.com/robinjia/adversarial-squad

Paraphrase - from Mohit Iyyer et al.'s work: https://github.com/miyyer/scpn

WordReplace - from Yuan Zang et al.'s work: https://github.com/thunlp/SememePSO-Attack



## Training and Evaluation 

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

## Reference

Please consider citing our work if you found this code or our paper beneficial to your research.

```
@inproceedings{Si2020BenchmarkingRO,
  title={Benchmarking Robustness of Machine Reading Comprehension Models},
  author={Chenglei Si and Ziqing Yang and Yiming Cui and Wentao Ma and Ting Liu and Shijin Wang},
  booktitle={Findings of ACL},
  year={2021},
}
```


If you encounter any problems, feel free to raise them in issues or contact the authors.


