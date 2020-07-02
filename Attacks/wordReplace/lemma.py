import pickle
import json
import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

with open('cache/tokenized_race_test_context.pkl', 'rb') as fh:
    test_data = pickle.load(fh)
with open('cache/race_test_tokenizer_v20k.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
dict = {w: i for (w, i) in tokenizer.word_index.items()}
inv_dict = {i: w for (w, i) in dict.items()}

word_candidate={}
test_text = [[inv_dict[t] for t in tt] for tt in test_data]

NNS={}
NNPS={}
JJR={}
JJS={}
RBR={}
RBS={}
VBD={}
VBG={}
VBN={}
VBP={}
VBZ={}
inv_NNS={}
inv_NNPS={}
inv_JJR={}
inv_JJS={}
inv_RBR={}
inv_RBS={}
inv_VBD={}
inv_VBG={}
inv_VBN={}
inv_VBP={}
inv_VBZ={}
s_ls=['NNS','NNPS','JJR','JJS','RBR','RBS','VBD','VBG','VBN','VBP','VBZ']
s_noun=['NNS','NNPS']
s_verb=['VBD','VBG','VBN','VBP','VBZ']
s_adj=['JJR','JJS']
s_adv=['RBR','RBS']
f=open('cache/race_test_pos_tags.pkl','rb')
all_pos_tag=pickle.load(f)
for idx in tqdm(range(len(test_text))):
    # print(idx)
    #text=train_text[idx]
    pos_tags = all_pos_tag[idx]
    for i in range(len(pos_tags)):
        pair=pos_tags[i]
        if pair[1] in s_ls:
            if pair[1][:2]=='NN':
                w = wnl.lemmatize(pair[0],pos='n')
            elif pair[1][:2]=='VB':
                w = wnl.lemmatize(pair[0], pos='v')
            elif pair[1][:2]=='JJ':
                w = wnl.lemmatize(pair[0], pos='a')
            else:
                w = wnl.lemmatize(pair[0], pos='r')
            eval('inv_'+pair[1])[w]=pair[0]
            eval(pair[1])[pair[0]]=w
f=open('cache/race_test_lemma_dict.pkl','wb')
pickle.dump((NNS,NNPS,JJR,JJS,RBR,RBS,VBD,VBG,VBN,VBP,VBZ,inv_NNS,inv_NNPS,inv_JJR,inv_JJS,inv_RBR,inv_RBS,inv_VBD,inv_VBG,inv_VBN,inv_VBP,inv_VBZ),f)
