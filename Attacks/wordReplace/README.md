# Running instructions on RACE dataset

Note: the codes are adapted from Zang Yuan's repo on Sememe-PSO attack.
## Process Data and Train Model

- Generate Candidate Substitution Words
```bash
python gen_pos_tag.py
python lemma.py
python gen_candidates.
```


## Craft Adversarial Examples
- Crafting Adversarial Examples for BERT
```bash
python AD_dpso_sem_bert.py
```
Note that we use rather small population size and max_iter, and simplified the population initialization process. This is to keep the running time under control. If you have more time and compute resources, you can try larger parameters.
