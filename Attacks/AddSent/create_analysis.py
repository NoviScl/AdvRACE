## create analysis test data
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


with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
	data = json.load(f)

PQA_dict = {}
for race_id, eg in tqdm(data.items()):
	eg_dict = {}
	qn = eg['question']
	context = eg['context']
	label = int(eg['label'])
	answer = eg['options'][label]

	distract = qn + ' ' + answer 
	context = nltk.sent_tokenize(context)

	pos = random.randint(0, len(context))
	context.insert(pos, distract)
	context = ' '.join(context)

	eg_dict['question'] = qn 
	eg_dict['context'] = context 
	eg_dict['options'] = eg['options']
	eg_dict['label'] = eg['label']

	PQA_dict[race_id] = eg_dict 

with open('analysis/test_dis.json', 'w') as f:
	json.dump(PQA_dict, f, indent=4)



