import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import pickle
from time import time

from pso_bert_min2 import PSOAttack

from torch.autograd import Variable
import torch
import torch.nn as nn
import json 
import os
from tqdm import tqdm
import logging 
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#from model_nli import Model
from encap_race_roberta import Model

model_path = '/ps2/intern/clsi/output_race_roberta/checkpoint-10000'
model = Model(model_path)

with open('cache/race_test_tokenizer_v20k.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}

with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
    test_data = json.load(f)

# for k,v in test_data.items():
#     logger.info('START')
#     logger.info(model(v))
#     break

with open('cache/race_test_pos_tags.pkl','rb') as fp:
    test_pos_tags=pickle.load(fp)
with open('cache/race_test_word_candidates_sense.pkl','rb') as f:
    word_candidate=pickle.load(f)

# with open('./all_seqs.pkl', 'rb') as fh:
#     train,valid,test = pickle.load(fh)
# with open('pos_tags_test.pkl','rb') as fp:
#     test_pos_tags=pickle.load(fp)
# test_s1=[t[1:-1] for t in test['s1']]
# test_s2=[t[1:-1] for t in test['s2']]
# #model=Model()
#model.evaluate(test['s1'],test['s2'],test['label'])
# np.random.seed(3333)
# vocab = {w: i for (w, i) in tokenizer.word_index.items()}
# inv_vocab = {i: w for (w, i) in vocab.items()}
# model = Model(inv_vocab) 

#test_accuracy = model.evaluate([test[0], test[1]], test[2])[1]
#print('\nTest accuracy = ', test_accuracy)

# adversary = PSOAttack(model, tokenizer, word_candidate, pop_size=60, max_iters=20)
adversary = PSOAttack(model, tokenizer, word_candidate, pop_size=10, max_iters=5)
# print ('the length of test cases is:', len(list(test_data)))
# TEST_SIZE = 5000
# test_idxs = np.random.choice(len(test_s1), size=TEST_SIZE, replace=False)
PQA_dict = {}
# test_list = []
# input_list = []
# output_list = []
# dist_list = []
# test_times = []
# success=[]
# change_list=[]
# target_list=[]
# true_label_list=[]
# success_count = 0
# i = 0
# while len(test_list) < 1000:
#     print('\n')
success_count = 0 
total_test = len(list(test_data))
test_idx = -1
for race_id, eg in tqdm(test_data.items()):
    test_idx += 1
    # if test_idx >= 100:
    #     break
    eg_dict = {}
    eg_dict['question'] = eg['question']
    eg_dict['options'] = eg['options']
    eg_dict['label'] = eg['label']
    # test_idx = test_idxs[i]
    # s1=test_s1[test_idx]
    # s2=test_s2[test_idx]
    context_orig = tokenizer.texts_to_sequences([eg["context"]])[0]
    # logger.info(str(len(context_orig)))
    # print (context_orig)
    pos_tags = test_pos_tags[test_idx]
    ### clsi: we will do the attack to all examples, including correct ones
    target = int(eg['label']) ## untargeted attack, target is to be different from true label
    start_time = time()
    attack_result = adversary.attack(context_orig, eg, target, pos_tags) ## will return the altered context
    if attack_result is None:
        # print('**** Attack failed **** ')
        eg_dict['context'] = eg['context']
    else:
        attack_result = attack_result[1]
        if len(attack_result) != len(context_orig):
            print ('Length Mismatch!')

        num_changes = np.sum(np.array(context_orig) != np.array(attack_result))
        x_len = np.sum(np.sign(context_orig))

        print('%d - %d changed.' % (test_idx, int(num_changes)))
        modify_ratio = num_changes / x_len
        if modify_ratio > 0.25:
            # logger.info('too much modify: '+str(modify_ratio))
            eg_dict['context'] = eg['context']
        else:
            # logger.info('***** SUCCESS: '+str(test_idx))
            success_count += 1
            new_context = ' '.join([inv_vocab[t] for t in attack_result])
            eg_dict['context'] = new_context 
            # print('***** DONE ', len(test_list), '------')
            # test_times.append(time() - start_time)
            # true_label_list.append(true_label)
            # input_list.append([s1, s2, true_label])
            # output_list.append(attack_result)
            # success.append(test_idx)
            # target_list.append(target)
            # change_list.append(modify_ratio)
    PQA_dict[race_id] = eg_dict

logger.info('Success rate: '+str(success_count / len(list(test_data))))
# f = open('AD_dpso_sem_bert.pkl', 'wb')
# pickle.dump((true_label_list, output_list, success, change_list, target_list), f)
with open('SemPSO_test_dis.json', 'w') as f:
    json.dump(PQA_dict, f, indent=4)
