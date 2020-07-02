import pickle
import spacy
import json
from tqdm import tqdm 
import random
random.seed(2020)

from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()

nlp = spacy.load("en_core_web_sm")

tag_dict = {
	'j': 'adj',
	'n': 'noun',
	'v': 'verb',
	'r': 'adv'
}

with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
	data_test = json.load(f)

with open('cache/test_candidates.pkl', 'rb') as fh:
    candidates = pickle.load(fh)

with open('cache/test_vocab.pkl', 'rb') as fh:
	vocab = pickle.load(fh)

PQA_dict = {}

'''
change all possible words will cause too much replacement,
so we set a threshold by the number of candidates available
'''
total_words = 0
total_changed = 0
for race_id, eg in tqdm(data_test.items()):
	eg_dict = {}
	eg_dict['question'] = eg['question']
	eg_dict['options'] = eg['options']
	eg_dict['label'] = eg['label']
	context = nlp(eg["context"])
	new_context = [tok.text for tok in context]
	total_words += len(context)
	total_words += len(eg["question"].split())
	for opt in eg["options"]:
		total_words += len(opt.split())
	num_cands = []
	for tok in context:
		if tok.lemma_ in candidates and tok.tag_[0].lower() in tag_dict:
			num_cands.append(len(candidates[tok.lemma_][tag_dict[tok.tag_[0].lower()]])) 
	num_cands.sort() 
	threshold = num_cands[len(num_cands)//2]
	for i in range(len(context)):
		tok = context[i]
		if tok.lemma_ in candidates and tok.tag_[0].lower() in tag_dict:
			if len(candidates[tok.lemma_][tag_dict[tok.tag_[0].lower()]]) > threshold:
				new_context[i] = random.choice(candidates[tok.lemma_][tag_dict[tok.tag_[0].lower()]])
				total_changed += 1
	new_context = detokenizer.detokenize(new_context)
	new_context = ' '.join(new_context.split())
	eg_dict['context'] = new_context
	PQA_dict[race_id] = eg_dict 

print (total_changed, total_words, total_changed/total_words)

with open('replace_test_dis.json', 'w') as f:
	json.dump(PQA_dict, f, indent=4)
