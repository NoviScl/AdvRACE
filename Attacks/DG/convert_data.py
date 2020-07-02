# convert Cleaned Dataset into 1-1 txt format
import json 
import unicodedata
from tqdm import tqdm

## convert everything into dict with question, context, options, label
## order can be different as long as content is the same
# PQA_dict_orig = {}
# PQA_dict_new = {}
# all_data_dict = {} #all data with file id as key

# # read original 
# testset_orig = []
# with open('cleaned_distractors/race_test.json', 'r') as f:
# 	for line in f:
# 		d = json.loads(line)
# 		testset_orig.append(d)

# with open('cleaned_unilm_dis.json', 'r') as f:
# 	testset_generated = json.load(f)

# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
# 	train = json.load(f)
# for k,v in train.items():
# 	key = k.split('/')[-1]
	

# ## Restore orginal test example
# for sample in testset_orig:
# 	fid = str(sample['id']['file_id'])


# for data in testset_orig:
# 	qid = str(data['id']['file_id']) + '_' + str(data['id']['question_id'])
# 	if qid not in testset_generated:
# 		print ('Missing')

# total_test_q = 0
# out_test = 0

# test_file_ids = []
# qids = []
# with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
#   data = json.load(f)

# for k,eg in data.items():
#   id = k.split('/')[-1]
#   id = id.split('.')[0]
#   test_file_ids.append(id)
#   total_test_q += 1

# test_file_ids = list(set(test_file_ids))

# # restore YF Gao's split 
# with open('cleaned_distractors/race_test.json', 'r') as f:
#   for line in f:
#     d = json.loads(line)
#     print (' '.join(d['answer_text']))
	# for k,v in d.items():
	#   print (k)
	# break
#     fid = str(d['id']['file_id'])
#     if fid not in test_file_ids:
#       out_test += 1
#     q_id = str(d['id']['file_id'])+'_'+str(d['id']['question_id'])
#     qids.append(q_id)
# qids = list(set(qids))
# print ('Number of files in original test set: ', len(test_file_ids))
# print ('Number of questions in original test set: ', total_test_q)
# print ('Number of questions NOT in irginal test set: ', out_test)
# print ('Number of questions in new test set: ', len(qids))





# Data Cleaning Codes are taken from Devlin's BERT preprocessing codes
def _is_control(char):
	"""Checks whether `chars` is a control character."""
	# These are technically control characters but we count them as whitespace
	# characters.
	if char == "\t" or char == "\n" or char == "\r":
		return False
	cat = unicodedata.category(char)
	if cat in ("Cc", "Cf"):
		return True
	return False

def _is_whitespace(char):
	"""Checks whether `chars` is a whitespace character."""
	# \t, \n, and \r are technically contorl characters but we treat them
	# as whitespace since they are generally considered as such.
	if char == " " or char == "\t" or char == "\n" or char == "\r":
		return True
	cat = unicodedata.category(char)
	if cat == "Zs":
		return True
	return False

# def _run_strip_accents(text):
#   text = unicodedata.normalize("NFD", text)
#   output = []
#   for char in text:
#       cat = unicodedata.category(char)
#       if cat == "Mn":
#           continue
#       output.append(char)
#   return "".join(output)

def _clean_text(text):
	"""Performs invalid character removal and whitespace cleanup on text + Remove accents"""
	text = unicodedata.normalize("NFD", text)
	output = []
	for char in text:
		cp = ord(char)
		if cp == 0 or cp == 0xfffd or _is_control(char):
	  		continue
		cat = unicodedata.category(char)
		if cat == "Mn":
			continue
		if _is_whitespace(char):
	  		output.append(" ")
		else:
	  		output.append(char)
	return "".join(output)

# file_dev = open('cleaned_distractors/race_dev.json', 'r', encoding='utf-8')
# file_train = open('cleaned_distractors/race_train.json', 'r', encoding='utf-8')
# file_test = open('cleaned_distractors/race_test.json', 'r', encoding='utf-8')

# src_dev = open('data_seq2seq/src-dev.txt', 'w+')
# tgt_dev = open('data_seq2seq/tgt-dev.txt', 'w+')
# src_train = open('data_seq2seq/src-train.txt', 'w+')
# tgt_train = open('data_seq2seq/tgt-train.txt', 'w+')
# src_test = open('data_seq2seq/src-test.txt', 'w+')
# tgt_test = open('data_seq2seq/tgt-test.txt', 'w+')

# for line in tqdm(file_dev.readlines()):
#     dic = json.loads(line)
#     src = ' '.join(dic['article'] + dic['question'])
#     src = _clean_text(src)
#     tgt = ' '.join(dic['distractor'])
#     tgt = _clean_text(tgt)
#     src_dev.write(src+'\n')
#     tgt_dev.write(tgt+'\n')

# src_dev.seek(0)
# print (len(src_dev.readlines()))
# tgt_dev.seek(0)
# print (len(tgt_dev.readlines()))

# for line in tqdm(file_train.readlines()):
#     dic = json.loads(line)
#     src = ' '.join(dic['article'] + dic['question'])
#     src = _clean_text(src)
#     tgt = ' '.join(dic['distractor'])
#     tgt = _clean_text(tgt)
#     src_train.write(src+'\n')
#     tgt_train.write(tgt+'\n')

# src_train.seek(0)
# print (len(src_train.readlines()))
# tgt_train.seek(0)
# print (len(tgt_train.readlines()))

# for line in tqdm(file_test.readlines()):
#     dic = json.loads(line)
#     src = ' '.join(dic['article'] + dic['question'])
#     src = _clean_text(src)
#     tgt = ' '.join(dic['distractor'])
#     tgt = _clean_text(tgt)
#     src_test.write(src+'\n')
#     tgt_test.write(tgt+'\n')

# src_test.seek(0)
# print (len(src_test.readlines()))
# tgt_test.seek(0)
# print (len(tgt_test.readlines()))

'''
keys:
article: list of tokens of article
sent: list of lists. Each list contains tokens of a sentence.
question: tokens of question ('_' signs removed)
distractor: tokens of distractor
'''


