import json
import random 
import nltk 
from tqdm import tqdm 
random.seed(2020)

# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
# 	data = json.load(f)

data = {}

## constrcut random span distractor test set
with open('/ps2/intern/clsi/RACE/test_dis.json', 'r') as f:
	data_orig = json.load(f)

for race_id, eg in tqdm(data_orig.items()):
	options = eg['options']
	label = int(eg['label'])
	answer = options[label]
	ans_len = len(nltk.word_tokenize(answer))
	context_tok = nltk.word_tokenize(eg['context'])
	new_opt = []
	for i in range(3):
		start = random.randint(0, len(context_tok)-ans_len)
		opt = ' '.join(context_tok[start : start + ans_len])
		new_opt.append(opt)
	new_opt.insert(label, answer)
	eg_dict = {}
	eg_dict['question'] = eg['question']
	eg_dict['context'] = eg['context']
	eg_dict['options'] = new_opt 
	eg_dict['label'] = label 

	data[race_id] = eg_dict 

with open('/ps2/intern/clsi/extractive/random_span/test_dis.json', 'w') as f:
	json.dump(data, f, indent=4)



# ## PQ-remove for orig
# with open('/ps2/intern/clsi/RACE/train_dis.json', 'r') as f:
# 	data_orig = json.load(f)

# total_len = len(list(data_orig))
# if total_len > 50000:
# 	data_orig_k = random.sample(list(data_orig), int(total_len*0.25))

# for k in data_orig_k:
# 	data[k+'_orig'] = data_orig[k]
# 	data[k+'_orig']['context'] = ''
# 	data[k+'_orig']['question'] = ''


# print ('+orig, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/PQremove_orig/train_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)

# with open('/ps2/intern/clsi/RACE/test_dis.json', 'r') as f:
# 	data_orig = json.load(f)

# for k in list(data_orig):
# 	data[k+'_orig'] = data_orig[k]
# 	data[k+'_orig']['context'] = ''
# 	data[k+'_orig']['question'] = ''

# print ('+orig, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/PQremove_orig/test_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)

# ## PQ-remove for extractive
# with open('/ps2/intern/clsi/final_distractor_datasets/extractive_old/train_dis.json', 'r') as f:
# 	data_extract = json.load(f)

# total_len = len(list(data_extract))
# if total_len > 50000:
# 	data_extract_k = random.sample(list(data_extract), int(total_len*0.25))

# for k in data_extract_k:
# 	data[k+'_extract'] = data_extract[k]
# 	data[k+'_extract']['context'] = ''
# 	data[k+'_extract']['question'] = ''


# print ('+DE, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/PQremove_extractive/train_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)

# with open('/ps2/intern/clsi/final_distractor_datasets/extractive_fixed/test_dis.json', 'r') as f:
# 	data_ext = json.load(f)

# for k in list(data_ext):
# 	data[k+'_extract'] = data_ext[k]
# 	data[k+'_extract']['context'] = ''
# 	data[k+'_extract']['question'] = ''

# print ('+DE, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/PQremove_extractive/test_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)

# ## PQ-remove for unilm
# with open('/ps2/intern/clsi/unilm/train_25_dis.json', 'r') as f:
# 	data_unilm = json.load(f)

# for k in list(data_unilm):
# 	data[k+'_unilm'] = data_unilm[k]
# 	data[k+'_unilm']['context'] = ''
# 	data[k+'_unilm']['question'] = ''


# print ('+unilm, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/PQremove_unilm/train_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)

# with open('/ps2/intern/clsi/final_distractor_datasets/unilm_fixed/test_dis.json', 'r') as f:
# 	data_unilm_test = json.load(f)

# for k in list(data_unilm_test):
# 	data[k+'_unilm'] = data_unilm_test[k]
# 	data[k+'_unilm']['context'] = ''
# 	data[k+'_unilm']['question'] = ''

# print ('+unilm, ', len(list(data)))

# with open('/ps2/intern/clsi/extractive/PQremove_unilm/test_dis.json', 'w') as f:
# 	json.dump(data, f, indent=4)
