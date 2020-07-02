import json
import random 
random.seed(2020)

# data_all = {}

with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
	data = json.load(f)

# ## load AddSent
with open('/ps2/intern/clsi/AddSent/finalAddSent/train_dis.json', 'r') as f:
	data_addsent = json.load(f)

total_len = len(list(data_addsent))
if total_len > 50000:
	data_addsent_k = random.sample(list(data_addsent), int(total_len*0.25))

for k in data_addsent_k:
	data[k+'_addsent'] = data_addsent[k]

print ('+AddSent: ', len(list(data)))


## Sample Training Data
# data_new = {}
# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
# 	data = json.load(f)

# total_len = len(list(data))
# if total_len > 50000:
# 	data_k = random.sample(list(data), int(total_len*0.25))

# for k in data_k:
# 	data_new[k] = data[k]

# print (len(list(data_new)))

# with open('/ps2/intern/clsi/RACE/train_25.json', 'w') as f:
# 	json.dump(data_new, f, indent=4)

# ## load charSwap 
with open('/ps2/intern/clsi/charSwap/charSwap_train/train_dis.json', 'r') as f:
	data_charswap = json.load(f)

total_len = len(list(data_charswap))
if total_len > 50000:
	data_charswap_k = random.sample(list(data_charswap), int(total_len*0.25))

for k in data_charswap_k:
	data[k+'_charswap'] = data_charswap[k]

print ('+CharSwap: ', len(list(data)))

# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
# 	data = json.load(f)

# ## Sample 25% DE, combine for AdvTrain
with open('/ps2/intern/clsi/final_distractor_datasets/extractive_old/train_dis.json', 'r') as f:
	data_extract = json.load(f)

total_len = len(list(data_extract))
if total_len > 50000:
	data_extract_k = random.sample(list(data_extract), int(total_len*0.25))

for k in data_extract_k:
	data[k+'_extract'] = data_extract[k]

print ('+DE, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/adv_training/train_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)

# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
# 	data = json.load(f)

## Sample 25% unilm, combine for AdvTrain
with open('/ps2/intern/clsi/unilm/train_25_dis.json', 'r') as f:
	data_unilm = json.load(f)

# t_k = random.sample(list(data_extract), int(total_len*0.25))

for k,v in data_unilm.items():
	data[k+'_unilm'] = v 

print ('+DG: ', len(list(data)))

with open('/ps2/intern/clsi/final_distractor_datasets/all_attack_adv_train/train_dis.json', 'w') as f:
	json.dump(data, f, indent=4)
