import pickle
import json
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm

stopwords = stopwords.words('english')

nlp = spacy.load("en_core_web_sm")

with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
	data_test = json.load(f)

'''
the dict stores:
lemma: pos tag
'''
all_vocab = []

for race_id, eg in tqdm(data_test.items()):
	context = eg["context"]
	doc = nlp(context)
	for tok in doc:
		if tok.lemma_ in all_vocab:
			continue 
		if tok.text.lower() in stopwords:
			continue 
		## don't change named entities
		if len(tok.ent_type_) > 0:
			continue 
		if tok.tag_[0].lower() in 'vjnr':
			all_vocab.append(tok.lemma_)

print ('Number of vocab: ', len(all_vocab))

f=open('cache/test_vocab.pkl','wb')
pickle.dump(all_vocab, f)
f.close()

# test_text = [[inv_dict[t] for t in tt] for tt in tests]

# all_pos_tags = []

# for text in test_text:
#     pos_tags = pos_tagger.tag(text)
#     all_pos_tags.append(pos_tags)
# f = open('pos_tags_test.pkl', 'wb')
# pickle.dump(all_pos_tags, f)
# f.close()
