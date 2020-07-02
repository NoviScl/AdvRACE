## transform 

import argparse 
import os 
import json
import numpy as np
import nltk
import string 
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet as wn 
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm 
import random
random.seed(2020)
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
detokenizer = TreebankWordDetokenizer()
stopwords = stopwords.words('english')
import spacy 
nlp = spacy.load("en_core_web_sm")


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


total_n = 0
total_changed = 0
total_random = 0
total_ne = 0 #named entities / nouns 
total_arnv = 0 #changed to antonyms
total_no_antonym = 0

# dict: word -> postag
with open('postag_dict.json', 'r') as f:
	pos_dict = json.load(f)

def transform(sent, sim_dict):
	''' 
	function to transform a word, return a detokenized sentence
	Some of the codes are inspired by the adversarial squad work by robin & percy
	'''
	
	# tok_sent = nltk.word_tokenize(sent.lower())
	# pos_sent = nltk.pos_tag(tok_sent)
	# ne_chunk = nltk.ne_chunk(pos_sent)
		
	doc = nlp(sent)
	tok_sent = [tok.text for tok in doc]
	pos_sent = [tok.tag_ for tok in doc]
	# note: empty string for tokens that are not named entities
	ne_chunk = [tok.ent_type_ for tok in doc]

	assert len(tok_sent) == len(pos_sent) and len(tok_sent) == len(ne_chunk), 'spacy doc len wrong'
	
	new_sent = tok_sent[:]
	changed_antonym = False

	global total_n 
	global total_changed
	global total_ne 
	global total_arnv
	global total_random 
	global total_no_antonym
	
	for i in range(len(tok_sent)):
		word = tok_sent[i]
		pos = pos_sent[i]
		replacement = None 
		
		## don't change stopwords or punctuations
		if word.lower() in stopwords or word in string.punctuation:
			continue 
		## simple way to alter numbers: change its value
		## only took care of integers here because other numbers are very rare in questions
		if word.isdigit():
			w = str(int(word)+random.randint(10, 100))
			new_sent[i] = w 
			changed_antonym = True 

		## step 1: change named entity and nouns to a distant GloVe word
		elif len(ne_chunk[i])>0:
			if word.lower() in sim_dict:
				sim_list = sim_dict[word.lower()][1:]
				for wd in sim_list:
					w = wd['word']
					if w in stopwords or w in string.punctuation:
						continue 
					if stemmer.stem(w.lower()) == stemmer.stem(word.lower()):
						continue 
					## Check POS 
					if w in pos_dict and pos_dict[w] != pos:
						continue 
					## finalise replacement word and record
					new_sent[i] = w 
					total_ne += 1
					changed_antonym = True 
					break

		# step 2: change adv, adj, v to antonyms
		# we only change one word antonym to prevent confusions
		if pos[0].lower() in ['r', 'v', 'j'] and not changed_antonym:
			# randomly sample one word from the set
			if pos[0].lower() == 'j':
				pos_ = 'a'
			else:
				pos_ = pos[0].lower()

			antonym_words = []
			synsets = wn.synsets(word.lower(), pos=pos_)
			for syn in synsets:
				for lemma in syn.lemmas():
					if lemma.antonyms():
						antonym = lemma.antonyms()[0].name()
						if antonym in pos_dict and pos_dict[antonym] != pos:
							continue  
						antonym_words.append(antonym)
							
			if len(antonym_words)>0:
				antonym = random.choice(antonym_words)
				new_sent[i] = antonym 
				total_arnv += 1
				changed_antonym = True 
	
	total_n += 1	
	if new_sent == tok_sent:
		total_no_antonym += 1

	# if no antonym is available, add negation word
	negate_words = ['is', 'was', 'were', 'are', 'will', 'did', 'do', 'does', 'may', 'would', 'should', 'could', 'can']
	negate_types = ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	if new_sent == tok_sent and not changed_antonym:
		if 'not' in new_sent:
			new_sent.remove('not')
			changed_antonym = True 
	if  new_sent == tok_sent and not changed_antonym:
		for n in negate_words:
			if n in tok_sent:
				idx = tok_sent.index(n)
				new_sent.insert(idx+1, 'not')
				changed_antonym = True 
				break 
	if  new_sent == tok_sent and not changed_antonym:
		for t in negate_types:
			if t in pos_sent:
				idx = pos_sent.index(t)
				new_sent.insert(idx, 'not')
				changed_antonym = True 
				break 
	if new_sent != tok_sent:
		total_changed += 1

	new_sent = detokenizer.detokenize(new_sent)
	new_sent = ' '.join(new_sent.split())
	return new_sent

def main():
	total_words = 0 
	# open RACE data
	# with open('../RACE/test.json', 'r') as f:
	# 	data = json.load(f)
	# with open('/ps2/intern/clsi/final_distractor_datasets/extractive/test_dis.json', 'r') as f:
	# 	data = json.load(f)
	with open('/ps2/intern/clsi/final_distractor_datasets/extractive/test_dis.json', 'r') as f:
		data = json.load(f)
	
	# # load nearest beighbor dict
	with open('/ps2/intern/clsi/AddSent/nearby_glove100.json', 'r') as f:
		sim = json.load(f)

	PQA_dict = {}
	for race_id, eg in tqdm(data.items()):
		eg_dict = {}
		
		qn = eg["question"]
		if 'following' in qn.lower() and 'true' in qn.lower() and 'not' not in qn.lower():
			for i in range(4):
				if i == int(eg["label"]):
					continue 
				opt = eg["options"][i]
				# print (opt)
				opt = transform(opt, sim)
				# print (opt+'\n')	
				eg["options"][i] = opt		
		 
		eg_dict["question"] = qn 
		eg_dict["context"] = eg["context"]
		eg_dict["options"] = eg["options"]
		eg_dict["label"] = eg["label"]
		PQA_dict[race_id] = eg_dict 

	print ('Total changed: ', total_changed)
	print ('Total Questions: ', total_n)
	print ('Total NE: ', total_ne)
	print ('Total arnv: ', total_arnv)
	print ('Total no antonym: ', total_no_antonym)
	
	with open('test_dis_extractive_fixed.json', 'w') as f:
		json.dump(PQA_dict, f, indent=4)
	
if __name__=='__main__':
	main()
	



