import torch, time, sys, argparse, os, codecs, h5py, csv, json, nltk
import string, random
from nltk.corpus import stopwords 
import _pickle as cPickle
import numpy as np
from torch.autograd import Variable
from nltk import ParentedTree
from train_scpn import SCPN
from train_parse_generator import ParseNet
from subwordnmt.apply_bpe import BPE, read_vocabulary
from scpn_utils import deleaf, parse_indexify_transformations
from tqdm import tqdm 
import spacy 
from benepar.spacy_plugin import BeneparComponent 
import logging 
random.seed(2020)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
stopwords = stopwords.words('english')
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(BeneparComponent("benepar_en"))


## 10 frequent templates
templates = [
            '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( S ( VP ) ( . ) ) ) EOP',
            '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
            '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
            '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
            '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
            '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
    ]
     
# templates = []
# with open('race_test_template.txt', 'r') as f:
#     fl = f.readlines()
#     for line in fl:
#         templates.append(line.strip())
# print (templates)

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

    ROOT is already included in the parse.
    '''
    lst = deleaf(parse)
    level = 0
    stack = []
    for tok in lst:
        if tok == '(':
            level += 1
        if level <= 3:
            stack.append(tok)
        if tok == ')':
            level -= 1
    # stack.insert(0, 'ROOT')
    # stack.insert(0, '(')
    # stack.append(')')
    stack.append('EOP')
    return ' '.join(stack)

def get_parse(sent):
    doc = nlp(sent)
    sent = list(doc.sents)[0]
    parse = sent._.parse_string 
    parse = '(ROOT ' + str(parse) + ')'
    return parse 

def reverse_bpe(sent):
    x = []
    cache = ''

    for w in sent:
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
        else:
            x.append(w)

    return ' '.join(x)

# encode sentences and parses for targeted paraphrasing
def encode_data(ssent, parse):

    ## randomly choose a template and encode the template
    # cur_temp = [random.choice(templates)]
    temp = extract_temp(parse)
    # print (temp) 
    cur_temp = [temp]
    ## use its original template
    # p = get_parse(ssent)
    # parse = '(ROOT ' + p + ')'
    # cur_temp = [extract_temp(p)]
    try:
        template_lens = [len(x.split()) for x in cur_temp]
        np_templates = np.zeros((len(cur_temp), max(template_lens)), dtype='int32')
        for z, template in enumerate(cur_temp):
            np_templates[z, :template_lens[z]] = [parse_gen_voc[w] for w in cur_temp[z].split()]
        tp_templates = Variable(torch.from_numpy(np_templates).long().cuda())
        tp_template_lens = torch.from_numpy(np.array(template_lens, dtype='int32')).long().cuda()
    except:
        return ''

    # fn = ['idx', 'template', 'generated_parse', 'sentence']
    # ofile = codecs.open(out_file, 'w', 'utf-8')
    # out = csv.DictWriter(ofile, delimiter='\t', fieldnames=fn)
    # out.writerow(dict((x,x) for x in fn))

    # read parsed data
    # infile = codecs.open(args.parsed_input_file, 'r', 'utf-8')
    # inrdr = csv.DictReader(infile, delimiter='\t')

    # loop over sentences and transform them
    # for d_idx, ex in enumerate(inrdr):
    #stime = time.time()
    #ssent = ' '.join(ex['tokens'].split())
    seg_sent = bpe.segment(ssent.lower()).split()

    # write gold sentence
    # out.writerow({'idx': ex['idx'],
    #     'template':'GOLD', 'generated_parse':ex['parse'], 
    #     'sentence':reverse_bpe(seg_sent)})

    # encode sentence using pp_vocab, leave one word for EOS
    seg_sent = [pp_vocab[w] for w in seg_sent if w in pp_vocab]

    # add EOS
    seg_sent.append(pp_vocab['EOS'])
    torch_sent = Variable(torch.from_numpy(np.array(seg_sent, dtype='int32')).long().cuda())
    torch_sent_len = torch.from_numpy(np.array([len(seg_sent)], dtype='int32')).long().cuda()

    # encode parse using parse vocab
    try:
        parse_tree = ParentedTree.fromstring(parse.strip())
        parse_tree = deleaf(parse_tree)
        np_parse = np.array([parse_gen_voc[w] for w in parse_tree], dtype='int32')
    except:
        return ''
    torch_parse = Variable(torch.from_numpy(np_parse).long().cuda())
    torch_parse_len = torch.from_numpy(np.array([len(parse_tree)], dtype='int32')).long().cuda()

    # generate full parses from templates
    beam_dict = parse_net.batch_beam_search(torch_parse.unsqueeze(0), tp_templates,
        torch_parse_len[:], tp_template_lens, parse_gen_voc['EOP'], beam_size=3, max_steps=150)
    seq_lens = []
    seqs = []
    for b_idx in beam_dict:
        prob,_,_,seq = beam_dict[b_idx][0]
        seq = seq[:-1] # chop off EOP
        seq_lens.append(len(seq))
        seqs.append(seq)
    np_parses = np.zeros((len(seqs), max(seq_lens)), dtype='int32')
    for z, seq in enumerate(seqs):
        np_parses[z, :seq_lens[z]] = seq
    tp_parses = Variable(torch.from_numpy(np_parses).long().cuda())
    tp_len = torch.from_numpy(np.array(seq_lens, dtype='int32')).long().cuda()
    

    # generate paraphrases from parses
    #try:
    beam_dict = net.batch_beam_search(torch_sent.unsqueeze(0), tp_parses, 
        torch_sent_len[:], tp_len, pp_vocab['EOS'], beam_size=3, max_steps=40)
    for b_idx in beam_dict:
        prob,_,_,seq = beam_dict[b_idx][0]
        #gen_parse = ' '.join([rev_label_voc[z] for z in seqs[b_idx]])
        gen_sent = ' '.join([rev_pp_vocab[w] for w in seq[:-1]])
        gen_sent  = gen_sent.replace('EOS', '')
        gen_sent = reverse_bpe(gen_sent.split())
        return gen_sent
        # out.writerow({'idx': ex['idx'],
        #     'template':templates[b_idx], 'generated_parse':gen_parse, 
        #     'sentence':reverse_bpe(gen_sent.split())})
    # except:
    #     print ('beam search OOM')

    # print (d_idx, time.time() - stime)


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='Syntactically Controlled Paraphrase Transformer')

    ## paraphrase model args
    parser.add_argument('--gpu', type=str, default='0',
            help='GPU id')
    parser.add_argument('--out_file', type=str, default='test_dis.json',
            help='paraphrase save path')
    parser.add_argument('--parsed_input_file', type=str, default='data/scpn_ex.tsv',
            help='parse load path')
    parser.add_argument('--input_file', type=str, default='/ps2/intern/clsi/RACE/test.json',
            help='original test json')
    parser.add_argument('--vocab', type=str, default='data/parse_vocab.pkl',
            help='word vocabulary')
    parser.add_argument('--parse_vocab', type=str, default='data/ptb_tagset.txt',
            help='tag vocabulary')
    parser.add_argument('--pp_model', type=str, default='models/scpn.pt',
            help='paraphrase model to load')
    parser.add_argument('--parse_model', type=str, default='models/parse_generator.pt',
            help='model save path')

    ## BPE args
    parser.add_argument('--bpe_codes', type=str, default='data/bpe.codes')
    parser.add_argument('--bpe_vocab', type=str, default='data/vocab.txt')
    parser.add_argument('--bpe_vocab_thresh', type=int, default=50)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    

    # load saved models
    pp_model = torch.load(args.pp_model)
    parse_model = torch.load(args.parse_model)

    # load vocab
    pp_vocab, rev_pp_vocab = cPickle.load(open(args.vocab, 'rb'))

    tag_file = codecs.open(args.parse_vocab, 'r', 'utf-8')
    parse_gen_voc = {}
    for idx, line in enumerate(tag_file):
        line = line.strip()
        parse_gen_voc[line] = idx
    rev_label_voc = dict((v,k) for (k,v) in parse_gen_voc.items()) 

    # load paraphrase network
    pp_args = pp_model['config_args']
    net = SCPN(pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans,
        len(pp_vocab), len(parse_gen_voc) - 1, pp_args.use_input_parse)
    net.cuda()
    net.load_state_dict(pp_model['state_dict'])
    net.eval()

    # load parse generator network
    parse_args = parse_model['config_args']
    parse_net = ParseNet(parse_args.d_nt, parse_args.d_hid, len(parse_gen_voc))
    parse_net.cuda()
    parse_net.load_state_dict(parse_model['state_dict'])
    parse_net.eval()

    # encode templates
    # template_lens = [len(x.split()) for x in templates]
    # np_templates = np.zeros((len(templates), max(template_lens)), dtype='int32')
    # for z, template in enumerate(templates):
    #     np_templates[z, :template_lens[z]] = [parse_gen_voc[w] for w in templates[z].split()]
    # tp_templates = Variable(torch.from_numpy(np_templates).long().cuda())
    # tp_template_lens = torch.from_numpy(np.array(template_lens, dtype='int32')).long().cuda()

    # instantiate BPE segmenter
    bpe_codes = codecs.open(args.bpe_codes, encoding='utf-8')
    bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

    # transform and store each PQA sample
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    with open('paraRACE_train_dis_25.json', 'r') as f:
        data_prev = json.load(f)
    data_prev = list(data_prev)
    ## sample more to complete the 16% previously sample
    ## process the rest 75% data for train
    data_keys = list(data)
    data_total = len(data_keys)
    for k in data_keys:
        if k in data_prev:
            data_keys.remove(k)
    if data_total > 10000:
        sample_keys = random.sample(data_keys, int(data_total*0.25))
        # sample_keys = data_keys
        new_data = {}
        for k in sample_keys:
            new_data[k] = data[k]
        data = new_data
    logger.info('Number of questions in total: '+str(len(data.items())))
    PQA_lst = {} #dict of dicts, keys are file_ids
    PQA_dict = {}
    total_sents = 0
    changed_sents = 0
    for race_id, eg in data.items():
        file_id = race_id.split('/')[-1].split('.')[0]
        if file_id not in PQA_lst:
            PQA_lst[file_id] = {}
        PQA_lst[file_id][race_id] = eg 

    for file_id, eg_dict in tqdm(PQA_lst.items()):
        qn_lst = []
        ans_lst = []
        keywords = []
        for race_id, eg in eg_dict.items():
            context = eg["context"]
            qn_lst.extend(nltk.word_tokenize(eg["question"].lower()))
            ans = eg["options"][int(eg["label"])]
            ans_lst.extend(nltk.word_tokenize(ans.lower()))
        for w in qn_lst+ans_lst:
            if w not in stopwords and w not in string.punctuation:
                keywords.append(w) 
        keywords = list(set(keywords))
        # print (keywords)
        context_sents = nltk.sent_tokenize(context)
        total_sents += len(context_sents)
        for i in range(len(context_sents)):
            sent = context_sents[i]
            for w in keywords:
                if w in sent.lower():
                    try:
                        new_sent = encode_data(sent, get_parse(sent))
                        if len(new_sent) > 0:
                            new_sent = list(new_sent)
                            new_sent[0] = new_sent[0].upper()
                            new_sent = "".join(new_sent)
                            # print (sen)
                            # print (new_sent+'\n')
                            context_sents[i] = new_sent 
                            changed_sents += 1
                        break 
                    except:
                        logger.info("Skipped one example.")
        new_context = ' '.join(context_sents)
        # print (new_context)
        for race_id, eg in eg_dict.items():
            eg_dict = {}
            eg_dict["question"] = eg["question"]
            eg_dict["context"] = new_context
            eg_dict["options"] = eg["options"]
            eg_dict["label"] = eg["label"]
            PQA_dict[race_id] = eg_dict 



    # PQA_dict = {}
    # total_sents = 0
    # changed_sents = 0
    # for race_id, eg in tqdm(data.items()):
    #     qn = eg["question"]
    #     options = eg["options"]
    #     context = eg["context"]
    #     context_sents = nltk.sent_tokenize(context)
    #     total_sents += len(context_sents)
    #     keywords = []
    #     options.append(qn)
    #     for opt in options:
    #         for w in nltk.word_tokenize(opt.lower()):
    #             if w not in string.punctuation and w not in stopwords:
    #                 keywords.append(w)
    #     keywords = list(set(keywords))
        # for i in range(len(context_sents)):
        #     sent = context_sents[i]
        #     sent_w = nltk.word_tokenize(sent.lower())
        #     sents = []
        #     for s in sent_w:
        #         if ',' in s:
        #             sents.extend(s.split(','))
        #         elif ':' in s:
        #             sents.extend(s.split(':'))
        #         elif ';' in s:
        #             sents.extend(s.split(';'))
        #         else:
        #             sents.append(s)
        # sents = []
        # for i in range(len(context_sents)):
        #     s = context_sents[i]
        #     # if ',' in s:
        #     #     sents.extend(s.split(','))
        #     # elif ':' in s:
        #     #     sents.extend(s.split(':'))
        #     if ';' in s:
        #         sents.extend(s.split(';'))
        #     else:
        #         sents.append(s)
        ## print (sents)
        # new_qn = encode_data(qn, get_parse(qn))
        # if len(new_qn) > 0:
        #     new_sent = list(new_qn)
        #     new_sent[0] = new_sent[0].upper()
        #     new_sent = "".join(new_sent)
        #     print (qn) 
        #     print (new_sent)
        # for sen in sents:
        #     for w in keywords:
        #         if w.lower() in sen.lower():
        #             # new_sent = encode_data(' '.join(nltk.word_tokenize(sent)), get_parse(sent))
        #             new_sent = encode_data(sen, get_parse(sen))
        #             if len(new_sent) > 0:
        #                 new_sent = list(new_sent)
        #                 new_sent[0] = new_sent[0].upper()
        #                 new_sent = "".join(new_sent)
        #                 print (sen)
        #                 print (new_sent+'\n')
        #                 # context_sents[i] = new_sent 
        #                 # changed_sents += 1
        #             break 
        # context = ' '.join(context_sents)
        # eg_dict = {}
        # eg_dict['question'] = qn
        # eg_dict['context'] = context 
        # eg_dict['options'] = eg['options']
        # eg_dict['label'] = eg['label']
        # PQA_dict[race_id] = eg_dict 

    with open(args.out_file, 'w') as f:
        json.dump(PQA_dict, f, indent=4)

    logger.info('Changed Sents Ratio: {}/{}={}'.format(changed_sents, total_sents, changed_sents/total_sents))


    # # paraphrase the sst!
    # encode_data(out_file=args.out_file)
