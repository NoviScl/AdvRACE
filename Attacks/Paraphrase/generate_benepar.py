import json, codecs, csv
import nltk
import spacy 
from benepar.spacy_plugin import BeneparComponent 
from tqdm import tqdm 
from collections import Counter

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(BeneparComponent("benepar_en"))

# eg_parse = '(PP (PP (VBG According) (PP (TO to) (NP (DT the) (NN passage)))) (, ,) (WHNP (WHNP (WDT which)) (PP (IN of) (NP (DT the) (NN following)))) (S (VP (VBZ is) (RB NOT) (ADJP (JJ true) (PP (IN about) (NP (NN football) (CC and) (NN rugby)))))) (. ?))'


def is_paren(tok):
    return tok == ")" or tok == "("

def deleaf(tree):
    nonleaves = ''
    for w in str(tree).replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split() 


def extract_temp(parse):
	'''
	The input is a parse string without the ROOT at the beginning.
	We extract the top 2 levels of the parse.
	'''
	lst = deleaf(parse)
	level = 0
	stack = []
	for tok in lst:
		if tok == '(':
			level += 1
		if level <= 2:
			stack.append(tok)
		if tok == ')':
			level -= 1
	stack.insert(0, 'ROOT')
	stack.insert(0, '(')
	stack.append(')')
	stack.append('EOP')
	return ' '.join(stack)

# print (extract_temp(eg_parse))


def get_parse(sent):
	doc = nlp(sent)
	sent = list(doc.sents)[0]
	parse = sent._.parse_string 
	#parse = '(ROOT ' + parse + ')'
	return parse 

all_temp = []
# get most frequent templates of questions
with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
	data = json.load(f)
for race_id, eg in tqdm(data.items()):
	passage = eg["context"]
	sents = nltk.sent_tokenize(passage)
	for sent in sents:
		parse = get_parse(sent)
		temp = extract_temp(parse)
		all_temp.append(temp)

tempDict = Counter(all_temp)
common = tempDict.most_common(10)

with open('race_test_template.txt', 'w+') as f:
	for t in common:
		print (t)
		f.write(t[0]+'\n')


# # #generate parses for questions
# fn = ['idx', 'tokens', 'parse']
# # file = open('data/trial.tsv', 'w')
# # out = csv.DictWriter(file, delimiter='\t', fieldnames=fn)
# # out.writerow(dict((x,x) for x in fn))

# with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
# 	data = json.load(f)

# with open('data/trial.tsv', 'w') as f:
# 	writer = csv.DictWriter(f, delimiter='\t', fieldnames=fn) 
# 	writer.writeheader()
# 	idx = 0
# 	for race_id, eg in data.items():
# 		if idx >= 20:
# 			break
# 		qn = eg["question"]
# 		if '_' in qn:
# 			qn = qn.replace('_', ' ')
# 		qn = ' '.join(nltk.word_tokenize(qn))
# 		parse = get_parse(qn)
# 		data = {}
# 		data['idx'] = idx
# 		data['tokens'] = qn
# 		data['parse'] = parse
# 		# print (parse)
# 		writer.writerow(data)
# 		idx += 1
		
