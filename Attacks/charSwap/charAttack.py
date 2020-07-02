#Updated version for char-level attack.
import json
import string
import random
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem.lancaster import LancasterStemmer
from tqdm import tqdm 
stemmer = LancasterStemmer()
detokenizer = TreebankWordDetokenizer()
random.seed(2020)

stopwords = stopwords.words('english')
letters = 'abcdefghijklmnopqrstuvwxyz'
words_total = 0
words_edited = 0
min_len = 4

keys = {
	'q': '12wsa',
	'w': 'q23edsa',
	'e': 'w34rfds',
	'r': 'e45tgfd',
	't': 'r56yhgf',
	'y': 'gt67ujh',
	'u': 'y78ikjh',
	'i': 'ju89olk',
	'o': 'i90plk',
	'p': "o0l-[';",
	'a': 'qwsxz',
	's': 'awedcxz',
	'd': 'serfvcx',
	'f': 'drtgvc',
	'g': 'ftyhbv',
	'h': 'gyujnb',
	'j': 'huikmn',
	'k': 'jiolm,',
	'l': "kop;.,",
	'z': 'asx',
	'x': 'zsdc',
	'c': 'xdfv',
	'v': 'cfgb',
	'b': 'vghn',
	'n': 'bhjm',
	'm': 'njk,'
}


def charSwap(text):
	global words_total
	global words_edited
	if len(text)<min_len:
		return text 

	if text.lower() in stopwords:
		return text 
	r = random.randint(1, len(text)-3)
	# try one more time 
	if text[r] == text[r+1]:
		r = random.randint(1, len(text)-3)
	if text[r] == text[r+1]:
		return text
	if text[r].isdigit() or text[r+1].isdigit():
		return text 
	if text[r] in string.punctuation or text[r+1] in string.punctuation:
		return text 
	text_ = list(text)
	text_[r], text_[r+1] = text_[r+1], text_[r]
	text_ = ''.join(text_)
	if text_ != text:
		words_edited += 1
	return text_


# def charSwap(text):
# 	if len(text)<min_len:
# 		return text 
# 	#strip punctuation, e.g. hello, 'football'
# 	if text[0] in string.punctuation:
# 		starter = text[0]
# 		text = text[1:]
# 	else:
# 		starter = ''
# 	if text[-1] in string.punctuation:
# 		ender = text[-1]
# 		text = text[:-1]
# 	else:
# 		ender = ''

# 	# if text[0].isupper():
# 	# 	return starter + text + ender
# 	if len(text)<min_len:
# 		return starter + text + ender 
# 	if text.lower() in stopwords:
# 		return starter + text + ender
# 	r = random.randint(1, len(text)-3)
# 	#'abba' is not possible to swap
# 	if text[r] == text[r+1]:
# 		r = random.randint(1, len(text)-3)
# 	text = list(text)
# 	text[r], text[r+1] = text[r+1], text[r]
# 	text = ''.join(text)
# 	return starter + text + ender

# def charDrop(text):
# 	if len(text)<min_len:
# 		return text 
# 	#strip punctuation, e.g. hello, 'football'
# 	if text[0] in string.punctuation:
# 		starter = text[0]
# 		text = text[1:]
# 	else:
# 		starter = ''
# 	if text[-1] in string.punctuation:
# 		ender = text[-1]
# 		text = text[:-1]
# 	else:
# 		ender = ''

# 	# if text[0].isupper():
# 	# 	return starter + text + ender
# 	if len(text)<min_len:
# 		return starter + text + ender 
# 	if text.lower() in stopwords:
# 		return starter + text + ender
# 	r = random.randint(1, len(text)-2)
	
# 	text = list(text)
# 	text = text[:r] + text[r+1:]
# 	text = ''.join(text)
# 	return starter + text + ender

# def charAdd(text):
# 	if len(text)<min_len:
# 		return text 
# 	#strip punctuation, e.g. hello, 'football'
# 	if text[0] in string.punctuation:
# 		starter = text[0]
# 		text = text[1:]
# 	else:
# 		starter = ''
# 	if text[-1] in string.punctuation:
# 		ender = text[-1]
# 		text = text[:-1]
# 	else:
# 		ender = ''

# 	# if text[0].isupper():
# 	# 	return starter + text + ender
# 	if len(text)<min_len:
# 		return starter + text + ender 
# 	if text.lower() in stopwords:
# 		return starter + text + ender
# 	r = random.randint(0, len(text)-2)
# 	c = random.choice(list(letters))
	
# 	text = list(text)
# 	text = text[:r] + [text[r], c] +  text[r+1:]
# 	text = ''.join(text)
# 	return starter + text + ender

# def charKey(text):
# 	if len(text)<min_len:
# 		return text 
# 	#strip punctuation, e.g. hello, 'football'
# 	if text[0] in string.punctuation:
# 		starter = text[0]
# 		text = text[1:]
# 	else:
# 		starter = ''
# 	if text[-1] in string.punctuation:
# 		ender = text[-1]
# 		text = text[:-1]
# 	else:
# 		ender = ''

# 	# if text[0].isupper():
# 	# 	return starter + text + ender
# 	if len(text)<min_len:
# 		return starter + text + ender 
# 	if text.lower() in stopwords:
# 		return starter + text + ender
# 	r = random.randint(1, len(text)-2)
# 	if text[r] not in keys:
# 		return starter + text + ender
# 	cands = list(keys[text[r]])
# 	new_c = random.choice(cands)

# 	text = list(text)
# 	text = text[:r] + [new_c] + text[r+1:]
# 	text = ''.join(text)
# 	return starter + text + ender

# def transform(sent):
# 	global words_total
# 	global words_edited
# 	word_list = sent.strip().split()
# 	#words_total += len(word_list)
# 	new_word_list = []
# 	for w in word_list:
# 		attack = random.randint(1,4)
# 		if attack == 1:
# 			new_w = charSwap(w)
# 		elif attack == 2:
# 			new_w = charKey(w)
# 		elif attack == 3:
# 			new_w = charDrop(w)
# 		else:
# 			new_w = charAdd(w)
# 		new_word_list.append(new_w)
# 		if new_w != w:
# 			words_edited += 1
		
# 	return ' '.join(new_word_list)


# def transform_sent(sent):
# 	sent = nltk.word_tokenize(sent)
# 	for i in range(len(sent)):
# 		rand = random.randint(0, len(sent)-1)
# 		if len(sent[i])<min_len or sent[i].lower() in stopwords:
# 			continue 
# 		w = charSwap(sent[i])
# 		if w!=sent[i]:
# 			sent[i] = w 
# 			break 
# 	return detokenizer.detokenize(sent)

def transform_sent(sent):
	# function to transform a word
	# return a detokenized sentence
	tok_sent = nltk.word_tokenize(sent)
	pos_sent = nltk.pos_tag(tok_sent)
	ne_chunk = nltk.ne_chunk(pos_sent)
	named_entity = []
	for tree in ne_chunk:
		if hasattr(tree, 'label'):
			for lf in tree.leaves():
				named_entity.append(lf[0])
	named_entity = list(set(named_entity))

	new_word_list = []
	global words_total
	global words_edited

	for i in range(len(pos_sent)):
		word = pos_sent[i][0]
		pos = pos_sent[i][1]
		
		if word.lower() in stopwords or word in string.punctuation:
			continue 
		if word in named_entity:
			continue 

		if pos[0].lower() in ['a', 'r', 'n', 'v']:
			word = charSwap(word)
			tok_sent[i] = word

	return detokenizer.detokenize(tok_sent)


def main():
	global words_total
	global words_edited 
	with open('../RACE/train_dis.json', 'r') as f:
		data = json.load(f)
	# with open('/ps2/intern/clsi/final_distractor_datasets/paraphrase/test_dis.json', 'r') as f:
	# 	data = json.load(f)
	# with open('/ps2/intern/clsi/AddSent/test_dis_CharSwapNoQn_AddSent.json', 'r') as f:
	# 	data = json.load(f)
	
	PQA_dict = {}
	for race_id, eg in tqdm(data.items()):
		eg_dict = {}
		question = eg["question"]
		context = eg["context"]
		options = eg["options"]
		label = eg["label"]
		answer = options[int(label)]

		# question = transform_sent(question)

		# sents = nltk.sent_tokenize(context)
		# for s in range(len(sents)):
		# 	sents[s] = transform_sent(sents[s])
		# context = ' '.join(sents)

		## collect entity list
		entity_list = []
		qw = nltk.word_tokenize(question)
		#words = qw + nltk.word_tokenize(answer)
		words = qw[:] 
		for opt in options:
			words.extend(nltk.word_tokenize(opt))

		for w in words:
			if len(w)>=min_len and w.lower() not in stopwords:
				entity_list.append(stemmer.stem(w.lower()))
				entity_list.append(w.lower())
		entity_list = list(set(entity_list))

		cw = nltk.word_tokenize(context)
		words_total += (len(cw) + len(words))
		
		# #words_total += len(qw)
		# #words_total += len(cw)
		# # for opt in options:
		# # 	words_total += len(nltk.word_tokenize(opt))

		# # transform question, charSwap all 
		for i in range(len(qw)):
			qw[i] = charSwap(qw[i])
			# qw[i] = transform(qw[i])
		question = detokenizer.detokenize(qw)
		# question = transform_sent(question)
		# context_list = []
		# for sent in nltk.sent_tokenize(context):
		# 	sent = transform_sent(sent)
		# 	context_list.append(sent)
		# context = ' '.join(context_list)
	
		## the correct strategy
		new_context_words = []
		for w in cw:
			if len(w)>=min_len and w.lower() not in stopwords:
				for entity in entity_list:
					stem_w = stemmer.stem(w.lower())
					if entity in stem_w or entity in w.lower():
						w = charSwap(w)
						#w = transform(w)
						break 
			new_context_words.append(w)

		# context = detokenizer.detokenize(new_context_words)
		
		## charSwap all 
		# new_context_words = []
		# for w in cw:
		# 	w = charSwap(w)
		# 	new_context_words.append(w)

		# context = detokenizer.detokenize(new_context_words)

		# sents = find_sentences(context)
		# for s in range(len(sents)):
		# 	for w in entity_list:
		# 		if w.lower() in sents[s].lower():
		# 			sents[s] = transform(sents[s])
		# 			#print (w, sents[s])
		# 			break 
		# context = ' '.join(sents) 

		#transform question
		#question = transform(question)
		#context = transform(context)

		#save data
		eg_dict["question"] = question
		eg_dict["context"] = context 
		eg_dict["options"] = options 
		eg_dict["label"] = label 
		PQA_dict[race_id] = eg_dict 
	
	print ("Total number of words: ", words_total)
	print ("Number of words edited: ", words_edited)
	print ("Ratio: ", words_edited/words_total)
	with open('charSwap_allWords_train_dis.json', 'w') as f:
		json.dump(PQA_dict, f, indent=4)


if __name__ == '__main__':
	main()	


	
