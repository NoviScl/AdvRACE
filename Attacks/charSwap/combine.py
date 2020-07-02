import json
import random 
random.seed(2020)

with open('/ps2/intern/clsi/RACE/train_dis.json', 'r') as f:
	data = json.load(f)

# ## load charSwap 
# with open('/ps2/intern/clsi/charSwap/charSwap_train/train_dis.json', 'r') as f:
# 	data_charswap = json.load(f)
# 	charSwap_allWords_train_dis
with open('charSwap_allWords_train_dis.json', 'r') as f:
	data_charswap = json.load(f)

total_len = len(list(data_charswap))
if total_len > 50000:
	data_charswap_k = random.sample(list(data_charswap), int(total_len*0.25))
# data_charswap_k = list(data_charswap)

for k in data_charswap_k:
	data[k+'_charswap'] = data_charswap[k]

print ('+CharSwap: ', len(list(data)))

with open('/ps2/intern/clsi/charSwap/advTrain_25_charSwap_all/train_dis.json', 'w') as f:
	json.dump(data, f, indent=4)

# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
# 	data = json.load(f)

# # ## Sample 25% DE, combine for AdvTrain
# with open('/ps2/intern/clsi/final_distractor_datasets/extractive_old/train_dis.json', 'r') as f:
# 	data_extract = json.load(f)

# total_len = len(list(data_extract))
# if total_len > 50000:
# 	data_extract_k = random.sample(list(data_extract), int(total_len*0.25))

# for k in data_extract_k:
# 	data[k+'_extract'] = data_extract[k]

# print ('+DE, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/adv_training/train_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)

# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
# 	data = json.load(f)

## Sample 25% unilm, combine for AdvTrain
# with open('/ps2/intern/clsi/unilm/train_25_dis.json', 'r') as f:
# 	data_unilm = json.load(f)

# # t_k = random.sample(list(data_extract), int(total_len*0.25))

# for k,v in data_unilm.items():
# 	data[k+'_unilm'] = v 

# print ('+DG: ', len(list(data)))

# with open('/ps2/intern/clsi/final_distractor_datasets/all_attack_adv_train/train_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)
