from __future__ import division
import numpy as np

import tensorflow as tf
import copy
import random
random.seed(2020)

from tqdm import tqdm 
import logging 
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class PSOAttack(object):
    def __init__(self, model, tokenizer, candidate,
                 pop_size, max_iters):
        '''
        model: loaded RoBERTa model 
        '''
        self.candidate = candidate
        # self.dataset = dataset
        # self.dict = self.dataset.dict
        # self.inv_dict = self.dataset.inv_dict
        self.tokenizer = tokenizer 
        self.vocab = len(tokenizer.word_counts) + 1
        dict = {w: i for (w, i) in tokenizer.word_index.items()}
        self.dict = dict 
        self.inv_dict = {i: w for (w, i) in dict.items()}
        self.model = model
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.invoke_dict={}
        self.temp = 0.3
        self.invoke_time=0

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def predict(self, examples):
        '''
        examples: dict with P, Q, O
        '''
        # texts=[' '.join([self.inv_dict[t] for t in sentence if t!=0]) for sentence in sentences]
        # return self.model.predict(texts)
        return self.model(examples)

    def mutate(self, x_cur, w_select_probs, w_list):
        x_len = w_select_probs.shape[0]
        # print('w_select_probs:',w_select_probs)
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        return self.do_replace(x_cur, rand_idx, w_list[rand_idx])

    def generate_population(self, context_cur, eg_orig, neigbhours_list, target, pop_size, x_len, neighbours_len):
        tem = self.gen_h_score(context_cur, eg_orig, x_len, target, neighbours_len, neigbhours_list)
        x_orig = context_cur[:]
        if len(tem)==1:
            return tem ##successful altered context
        h_score, w_list=tem
        return [self.mutate(x_orig, h_score, w_list) for _ in
                range(pop_size)]

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def pred(self, adv_eg):
        
        if (adv_eg['context'], adv_eg['question']) not in self.invoke_dict:
            pred=self.predict(adv_eg)[0]
            self.invoke_dict[adv_eg['question']]=pred
            self.invoke_time+=1
            # logger.info('Doing Pred---')
        else:
            pred=self.invoke_dict[(adv_eg['context'], adv_eg['question'])]
       
        return pred

    def gen_most_change(self, pos, context_cur, eg_orig, target, replace_list):
        new_x_list = [self.do_replace(context_cur, pos, w) if context_cur[pos] != w and w != 0 else context_cur for w in replace_list]
        new_x_scores=[] 
        new_context_list = [' '.join([self.inv_dict[t] for t in tt]) for tt in new_x_list]
        for adv in new_x_list:
            ## retrieve context for model pred
            adv_context = ' '.join([self.inv_dict[t] for t in adv])
            new_eg = copy.deepcopy(eg_orig)
            new_eg['context'] = adv_context
            p=self.pred(new_eg)

            if np.argmax(p)!=target:
                return [adv] ## return the altered context ids

            p=p[target]
            new_x_scores.append(p)
        new_x_scores=np.array(new_x_scores)


        orig_score = self.pred(eg_orig)

        orig_score=orig_score[target]
        new_x_scores = orig_score-new_x_scores
        return [np.max(new_x_scores), new_x_list[np.argsort(new_x_scores)[-1]][pos]]

    def norm(self, n):

        tn = []
        for i in n:
            if i <= 0:
                tn.append(0)
            else:
                tn.append(i)
        s = np.sum(tn)
        if s == 0:
            for i in range(len(tn)):
                tn[i] = 1
            return [t / len(tn) for t in tn]
        new_n = [t / s for t in tn]

        return new_n

    def gen_h_score(self, context_cur, eg_orig, x_len, target, neighbours_len, neigbhours_list):
        x_now = context_cur[:]
        w_list = []
        prob_list = []
        # for i in range(x_len):
        #     if neighbours_len[i] == 0:
        #         w_list.append(x_now[i])
        #         prob_list.append(0)
        #         continue
        #     tem = self.gen_most_change(i, context_cur, eg_orig, target, neigbhours_list[i])

        #     if len(tem)==1:
        #         return tem
        #     p,w=tem
        #     w_list.append(w)
        #     prob_list.append(p)

        '''
        iterating through the entire candidate list if way too slow,
        use the number of candidates as approximation.
        Also, we will initialze with a random candidate for each position, instead of the best one.
        '''
        for i in range(x_len):
            if neighbours_len[i] == 0:
                w_list.append(x_now[i])
            else:
                w_list.append(random.choice(neigbhours_list[i]))
            prob_list.append(max(50, neighbours_len[i])) ##cap by 50

        prob_list = self.norm(prob_list)
        # print('neighbours_len:',neighbours_len)
        # print('prob_list:',prob_list)

        h_score = prob_list
        h_score = np.array(h_score)
        return [h_score, w_list]

    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(x != x_orig)) / float(x_len)
        return change_ratio

    def attack(self, context_orig, eg_orig, target, pos_tags):
        self.invoke_time = 0
        self.invoke_dict={}
        x_adv = context_orig.copy()
        x_len = np.sum(np.sign(context_orig))
        x_len = int(x_len)
        pos_list = ['JJ', 'NN', 'RB', 'VB']
        neigbhours_list = []
        for i in range(x_len):
            if x_adv[i] not in range(1, self.vocab):
                neigbhours_list.append([])
                continue
            pair = pos_tags[i]
            if pair[1][:2] not in pos_list:
                neigbhours_list.append([])
                continue
            if pair[1][:2] == 'JJ':
                pos = 'adj'
            elif pair[1][:2] == 'NN':
                pos = 'noun'
            elif pair[1][:2] == 'RB':
                pos = 'adv'
            else:
                pos = 'verb'
            if pos in self.candidate[x_adv[i]]:
                neigbhours_list.append([neighbor for neighbor in self.candidate[x_adv[i]][pos]])
            else:
                neigbhours_list.append([])

        neighbours_len = [len(x) for x in neigbhours_list]
        # logger.info('Number of candidate words: {}'.format(np.sum(neighbours_len)))
        orig_score=self.pred(eg_orig)
        # print('orig: ', orig_score[target]) ##for debugging

        if np.sum(neighbours_len) == 0:
            return None

        # print(neighbours_len)

        context_cur = context_orig[:]
        pop = self.generate_population(context_cur, eg_orig, neigbhours_list, target, self.pop_size, x_len, neighbours_len)

        # logger.info('Generated {} population.'.format(len(pop)))

        if len(pop)==1:
            return [self.invoke_time, pop[0]]
        part_elites = copy.deepcopy(pop)

        pop_scores=[]
        pop_scores_all=[]
        for a in pop:
            adv_context = ' '.join([self.inv_dict[t] for t in a])
            new_eg = copy.deepcopy(eg_orig)
            new_eg['context'] = adv_context
            pt=self.pred(new_eg)

            pop_scores.append(pt[target])
            pop_scores_all.append(pt)
        part_elites_scores = pop_scores
        all_elite_score = np.min(pop_scores)
        pop_ranks = np.argsort(pop_scores)
        top_attack = pop_ranks[0]
        all_elite = pop[top_attack]
        for pt_id in range(len(pop_scores_all)):
            pt=pop_scores_all[pt_id]
            if np.argmax(pt)!=target:
                # print('now',pt[target])
                return [self.invoke_time,pop[pt_id]]

        # logger.info('Start PSO.')

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for rrr in range(self.pop_size)]
        V_P = [[V[t] for rrr in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):
            # logger.info('Current Iter: '+str(i+1))

            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)


            for id in range(self.pop_size):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                                            all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2
                # P1=self.sigmod(P1)
                # P2=self.sigmod(P2)

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)

            # logger.info('Eval New Pop.')

            pop_scores = []
            pop_scores_all=[]
            for a in pop:
                adv_context = ' '.join([self.inv_dict[t] for t in a])
                new_eg = copy.deepcopy(eg_orig)
                new_eg['context'] = adv_context
                pt=self.pred(new_eg)

                pop_scores.append(pt[target])
                pop_scores_all.append(pt)
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[0]

            # print('\t\t', i, ' -- ', pop_scores[top_attack])
            for pt_id in range(len(pop_scores_all)):
                pt = pop_scores_all[pt_id]
                if np.argmax(pt) != target:

                    return [self.invoke_time, pop[pt_id]]

            # logger.info('Start Mutate.')

            new_pop = []
            for x in pop:
                change_ratio = self.count_change_ratio(x, context_orig, x_len)
                p_change = 1 - 2*change_ratio
                if np.random.uniform() < p_change:
                    tem= self.gen_h_score(x, eg_orig, x_len, target, neighbours_len, neigbhours_list)

                    if len(tem)==1:
                        return [self.invoke_time,tem[0]]
                    new_h, new_w_list=tem
                    new_pop.append(self.mutate(x, new_h, new_w_list))
                else:
                    new_pop.append(x)
            pop = new_pop

            # logger.info('Eval Mutate.')

            pop_scores = []
            pop_scores_all = []
            for a in pop:
                adv_context = ' '.join([self.inv_dict[t] for t in a])
                new_eg = copy.deepcopy(eg_orig)
                new_eg['context'] = adv_context
                pt=self.pred(new_eg)

                pop_scores.append(pt[target])
                pop_scores_all.append(pt)
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[0]

            for pt_id in range(len(pop_scores_all)):
                pt = pop_scores_all[pt_id]
                if np.argmax(pt) != target:
                    return [self.invoke_time, pop[pt_id]]
            for k in range(self.pop_size):
                if pop_scores[k] < part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            elite = pop[top_attack]
            if np.min(pop_scores) < all_elite_score:
                all_elite = elite
                all_elite_score = np.min(pop_scores)
        return None

