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

def transform_sent(sent, sim_dict):
	doc = nlp(sent)
	tok_sent = [tok.text for tok in doc]
	pos_sent = [tok.tag_ for tok in doc]
	ne_chunk = [tok.ent_type_ for tok in doc]

	new_sent = tok_sent[:]

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
			w = str(int(word)+random.randint(1, 10))
			new_sent[i] = w 
		## step 1: change named entity and nouns to a distant GloVe word
		elif len(ne_chunk[i])>0 or pos[0].lower() == 'n':
			if word.lower() in sim_dict:
				## reverse the order to pick from the farthest instead of random sampling
				## to maintain certain semantic difference
				sim_list = sim_dict[word.lower()][::-1]
				for wd in sim_list:
					w = wd['word']
					if w in stopwords or w in string.punctuation:
						continue 
					if stemmer.stem(w.lower()) == stemmer.stem(word.lower()):
						continue 
					## Check POS 
					if w not in pos_dict or pos_dict[w] != pos:
						continue 
					## finalise replacement word and record
					new_sent[i] = w 
					break
		# step 2: change adv, adj, v to antonyms
		elif pos[0].lower() in ['r', 'v', 'j']:
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
						if antonym not in pos_dict or pos_dict[antonym] != pos:
							continue  
						antonym_words.append(antonym)
							
			if len(antonym_words)>0:
				antonym = random.choice(antonym_words)
				new_sent[i] = antonym 

	## hard-coded fix, but very very rare (7 cases in RACE test set)
	if new_sent == tok_sent:
		new_sent.insert(0, 'not')
	new_sent = detokenizer.detokenize(new_sent)
	return ' '.join(new_sent.split())

def transform(sent, sim_dict, distractors, all_qn=None, all_qn_dict=None, all_fill=None, used_ne=None, used_antonym=None, used_distractor=None):
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

	if not used_ne:
		used_ne = []
	if not used_antonym:
		used_antonym = []
	if not used_distractor:
		used_distractor = []
	
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
					## use a difference word from previous time
					if w in used_ne:
						continue
					## Check POS 
					if w in pos_dict and pos_dict[w] != pos:
						continue 
					## finalise replacement word and record
					new_sent[i] = w 
					used_ne.append(w)
					total_ne += 1
					break

		# step 2: change adv, adj, v to antonyms
		# we only change one word antonym to prevent confusions
		elif pos[0].lower() in ['r', 'v', 'j'] and not changed_antonym:
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
			## use a different antonym if possible 
			if len(antonym_words) > 1 and used_antonym is not None:
				for w in used_antonym:
					if w in antonym_words:
						antonym_words.remove(w)
							
			if len(antonym_words)>0:
				antonym = random.choice(antonym_words)
				new_sent[i] = antonym 
				used_antonym.append(antonym)
				total_arnv += 1
				changed_antonym = True 
	
	total_n += 1	
	dis = random.choice(distractors) 
	used_distractor.append(dis)
	if new_sent != tok_sent:
		total_changed += 1

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
	## alter the distractor instead
	if new_sent == tok_sent:
		dis_orig = dis
		dis = transform_sent(dis, sim_dict)
		if dis_orig == dis:
			total_no_antonym += 1

	# if new_sent != tok_sent:
	# print (' '.join(tok_sent))
	# print (' '.join(new_sent)+'\n')
	# total_changed += 1
	# dis = random.choice(distractors)
	# while dis in used_distractor:
	# 	dis = random.choice(distractors)
	# used_distractor.append(dis)
	new_sent = detokenizer.detokenize(new_sent)
	if '_' in new_sent:
		new_sent = new_sent.replace('_', dis)
	else:
		new_sent = new_sent + ' ' + dis
	new_sent = ' '.join(new_sent.split())
	return new_sent, used_ne, used_antonym, used_distractor
	# else:
	# 	total_random += 1
	# 	if '_' in sent:
	# 		new_sent = random.choice(all_fill)
	# 		new_sent = new_sent.replace('_', dis)
	# 	else:
	# 		if tok_sent[0].lower() in all_qn_dict:
	# 			new_sent = random.choice(all_qn_dict[tok_sent[0].lower()])
	# 		else:
	# 			new_sent = random.choice(all_qn)
	# 		new_sent = new_sent + ' ' + dis
	# 	new_sent = ' '.join(new_sent.split())
	# 	return new_sent, used_ne, used_antonym, used_distractor


def main():
	total_words = 0 
	# open RACE data
	# with open('../RACE/train.json', 'r') as f:
	# 	data = json.load(f)
	# with open('/ps2/intern/clsi/final_distractor_datasets/paraphrase/test_dis.json', 'r') as f:
	# 	data = json.load(f)
	with open('/ps2/intern/clsi/charSwap/test_dis_charSwapNoQn.json', 'r') as f:
		data = json.load(f)
	
	# # load nearest beighbor dict
	with open('/ps2/intern/clsi/AddSent/nearby_glove100.json', 'r') as f:
		sim = json.load(f)

	'''
	collect all questions in test set first, 
	classify into question and fill-in, use first question word as key.
	'''
	all_qn_dict = {}
	all_qn = []
	all_fill = []
	for race_id, eg in data.items():
		qn = eg["question"]
		if '_' in qn:
			all_fill.append(qn)

		if '?' in qn:
			first_w = nltk.word_tokenize(qn)[0].lower()
			all_qn.append(qn)
			if first_w not in all_qn_dict:
				all_qn_dict[first_w] = [qn]
			else:
				all_qn_dict[first_w].append(qn)

	PQA_dict = {}
	for race_id, eg in tqdm(data.items()):
		eg_dict = {}
		
		qn = eg["question"]
		ans = eg["options"][int(eg["label"])]
		distractors = [eg["options"][i] for i in range(4) if i!=int(eg["label"])]
		context = eg["context"]
		
		dis_seq_1, used_ne, used_antonym, used_distractor = transform(qn, sim, distractors, all_qn, all_qn_dict, all_fill)
		# print (qn)
		# print (dis_seq_1+'\n')
		context = nltk.sent_tokenize(context)
		pos = random.randint(0, len(context))
		context.insert(pos, dis_seq_1)

		# second Q-D pair
		# if dis_seq_1 != '':
		for d in used_distractor:
			distractors.remove(d)
		dis_seq_2, used_ne, used_antonym, used_distractor = transform(qn, sim, distractors, all_qn, all_qn_dict, all_fill, used_ne, used_antonym, used_distractor)
		# print (qn)
		# print (dis_seq_2+'\n')
		pos = random.randint(0, len(context))
		context.insert(pos, dis_seq_2)

		# if dis_seq_1 != '':
		# 	# rand = random.randint(1, 2)
		# 	# if rand == 1:
		# 	# 	if '_' in question:
		# 	# 		question = question.replace('_', ans)
		# 	# 	else:
		# 	# 		question = question + ' ' + ans 
		# 	# else:
		# 	# 	if '_' in question:
		# 	# 		question = question.replace('_', random.choice(distractors))
		# 	# 	else:
		# 	# 		question = question + ' ' + random.choice(distractors)
		# 	# context = context + ' ' + question 
			
		# 	# if '_' in question:
		# 	# 	question = question.replace('_', random.choice(distractors))
		# 	# else:
		# 	# 	question = question + ' ' + random.choice(distractors)
			
		# 	# context = context + ' ' + question
		# 	context = nltk.sent_tokenize(context)
		# 	pos = random.randint(0, len(context))
		# 	context.insert(pos, question)
		context = ' '.join(context) 
		 
		eg_dict["question"] = qn 
		eg_dict["context"] = context 
		eg_dict["options"] = eg["options"]
		eg_dict["label"] = eg["label"]
		PQA_dict[race_id] = eg_dict 

	print ('Total changed: ', total_changed)
	print ('Total Random: ', total_random)
	print ('Total Questions: ', total_n)
	print ('Total NE: ', total_ne)
	print ('Total arnv: ', total_arnv)
	print ('Total no antonym: ', total_no_antonym)
	
	# with open('test_dis_CharSwapNoQn_AddSent.json', 'w') as f:
	# 	json.dump(PQA_dict, f, indent=4)
	# with open('AddSentRace/train_dis.json', 'w') as f:
	# 	json.dump(PQA_dict, f, indent=4)
	with open('test_dis_CharSwapNoQn_AddSent.json', 'w') as f:
		json.dump(PQA_dict, f, indent=4)
	
if __name__=='__main__':
	main()
	# 
	# # 
	# total_words = 0 
	# # open RACE data
	# with open('../RACE/test.json', 'r') as f:
	# 	data = json.load(f)
	# # with open('/ps2/intern/clsi/final_distractor_datasets/paraphrase/test_dis.json', 'r') as f:
	# # 	data = json.load(f)
	# # with open('/ps2/intern/clsi/charSwap/test_dis_charSwapNoQn.json', 'r') as f:
	# # 	data = json.load(f)
	
	# # # load nearest beighbor dict
	# with open('/ps2/intern/clsi/AddSent/nearby_glove100.json', 'r') as f:
	# 	sim = json.load(f)
	# question = 'Which of the following is TRUE about homeschooling according to the text?'
	# distractors = ["Homeschooling is still illegal in developed countries.",
 #            		"Athletes and actors can not be home-schooled.",
 #            		"There is no curriculum for homeschooled children.",]

	'''
	collect all questions in test set first, 
	classify into question and fill-in, use first question word as key.
	'''
	# all_qn_dict = {}
	# all_qn = []
	# all_fill = []
	# for race_id, eg in data.items():
	# 	qn = eg["question"]
	# 	if '_' in qn:
	# 		all_fill.append(qn)

	# 	if '?' in qn:
	# 		first_w = nltk.word_tokenize(qn)[0].lower()
	# 		all_qn.append(qn)
	# 		if first_w not in all_qn_dict:
	# 			all_qn_dict[first_w] = [qn]
	# 		else:
	# 			all_qn_dict[first_w].append(qn)

	# dis_seq_1, used_ne, used_antonym, used_distractor = transform(question, sim, distractors, all_qn, all_qn_dict, all_fill)
	# print (dis_seq_1)
	# dis_seq_2, used_ne, used_antonym, used_distractor = transform(question, sim, distractors, all_qn, all_qn_dict, all_fill, used_ne, used_antonym, used_distractor)
	# print (dis_seq_2)
	
	# synsets = wn.synsets('mainly', pos='r')
	# for syn in synsets:
	# 	for lemma in syn.lemmas():
	# 		if lemma.antonyms():
	# 			antonym = lemma.antonyms()[0].name()
				# print (antonym)
	# question = 'What is this passage mainly about?'
	# print (nltk.pos_tag(nltk.word_tokenize(question)))




