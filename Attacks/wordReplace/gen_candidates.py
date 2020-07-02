import pickle
import json
import nltk
from tqdm import tqdm
import OpenHowNet

word_candidate = {}

with open('cache/test_vocab.pkl', 'rb') as fh:
    vocab = pickle.load(fh)

hownet_dict = OpenHowNet.HowNetDict()

pos_list = ['noun', 'verb', 'adj', 'adv']
pos_set = set(pos_list)

word_pos = {}
word_sem = {}
for w in vocab:
    try:
        tree = hownet_dict.get_sememes_by_word(w, merge=False, structured=True, lang="en")
        w1_sememes = hownet_dict.get_sememes_by_word(w, structured=False, lang="en", merge=False)
        new_w1_sememes = [t['sememes'] for t in w1_sememes]
        # print(tree)

        w1_pos_list = [x['word']['en_grammar'] for x in tree]
        word_pos[w] = w1_pos_list
        word_sem[w] = new_w1_sememes
        main_sememe_list = hownet_dict.get_sememes_by_word(w, merge=False, structured=False, lang='en',
                                                           expanded_layer=2)
    except:
        word_pos[w] = []
        word_sem[w] = []
        main_sememe_list = []
    # assert len(w1_pos_list)==len(new_w1_sememes)
    # assert len(w1_pos_list)==len(main_sememe_list)


def add_w1(w):
    word_candidate[w] = {}
    # w1_s_flag = 0
    # w1_orig = None
    # w1_pos_sem = None

    # for s in s_ls:
    #     if w1 in eval(s):
    #         w1_s_flag = 1
    #         w1_pos_sem = s
    #         w1_orig = eval(s)[w1]
    #         break
    # if w1_s_flag == 0:
    #     w1_orig = w1
    #     w1_pos_sem = 'orig'

    w1_pos = set(word_pos[w])
    for pos in pos_set:
        word_candidate[w][pos] = []
    valid_pos_w1 = w1_pos & pos_set

    if len(valid_pos_w1) == 0:
        return

    new_w1_sememes = word_sem[w]
    if len(new_w1_sememes) == 0:
        return

    for w2 in vocab:

        if w == w2:
            continue
        # w2_s_flag = 0
        # w2_orig = None
        # w2_pos_sem = None
        # for s in s_ls:
        #     if w2 in eval(s):
        #         w2_s_flag = 1
        #         w2_pos_sem = s
        #         w2_orig = eval(s)[w2]
        #         break
        # if w2_s_flag == 0:
        #     w2_orig = w2
        #     w2_pos_sem = 'orig'

        w2_pos = set(word_pos[w2])
        all_pos = w2_pos & w1_pos & pos_set
        if len(all_pos) == 0:
            continue

        new_w2_sememes = word_sem[w2]
        # print(w2)
        # print(new_w1_sememes)
        # print(new_w2_sememes)
        if len(new_w2_sememes) == 0:
            continue
        # not_in_num1 = count(w1_sememes, w2_sememes)
        # not_in_num2 = count(w2_sememes,w1_sememes)
        # not_in_num=not_in_num1+not_in_num2
        w_flag=0

        for s1_id in range(len(new_w1_sememes)):
            if w_flag == 1:
                break
            pos_w1 = word_pos[w][s1_id]
            s1 = set(new_w1_sememes[s1_id])
            if pos_w1 not in pos_set:
                continue
            for s2_id in range(len(new_w2_sememes)):
                if w_flag==1:
                    break
                pos_w2 = word_pos[w2][s2_id]
                s2 = set(new_w2_sememes[s2_id])
                if pos_w1 == pos_w2 and s1 == s2:
                    # if w1_pos_sem == 'orig':
                    #     if w2_pos_sem == 'orig':
                    word_candidate[w][pos_w1].append(w2)
                    w_flag=1
                    break
                    # else:
                    #     for p in eval('s_' + pos_w1):
                    #         if w1 in eval(p) and w2 in eval(p):
                    #             word_candidate[i1][pos_w1].append(i2)
                    #             w_flag=1
                    #             break




for w in tqdm(vocab):

    # print(i1)

    add_w1(w)


f = open('cache/test_candidates.pkl', 'wb')
pickle.dump(word_candidate, f)
