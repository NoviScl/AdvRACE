"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import json
import random
import pickle
import nltk 
from eval.eval import eval 

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import seq2seq_loader as seq2seq_loader


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2))/len(s1.union(s2))

def loads_json(loadpath, loadinfo=None):
    with open(loadpath, 'r', encoding='utf-8') as fh:
        print (loadinfo)
        dataset = []
        for line in fh:
            example = json.loads(line)
            dataset.append(example)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument('--topk', type=int, default=10,
                        help="Value of K.")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")

    args = parser.parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # tokenizer = BertTokenizer.from_pretrained(
    #     args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer(vocab_file='/ps2/intern/clsi/BERT/bert_weights/cased_L-24_H-1024_A-16/vocab.txt', do_lower_case=args.do_lower_case)

    tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids,
                                                                    mode="s2s", num_qkv=args.num_qkv, s2s_special_token=args.s2s_special_token, s2s_add_segment=args.s2s_add_segment, s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
        (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])

    def _get_token_id_set(s):
        r = None
        if s:
            w_list = []
            for w in s.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            r = set(tokenizer.convert_tokens_to_ids(w_list))
        return r

    forbid_ignore_set = _get_token_id_set(args.forbid_ignore_word)
    not_predict_set = _get_token_id_set(args.not_predict_token)
    print(args.model_recover_path)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model, state_dict=model_recover, num_labels=cls_num_labels, num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                                      length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, not_predict_set=not_predict_set, ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode, max_position_embeddings=args.max_seq_length, 
                                                      ffn_type=args.ffn_type, num_qkv=args.num_qkv, seg_emb=args.seg_emb, pos_shift=args.pos_shift, topk=args.topk, config_path=args.config_path)
        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length

        ## for YFG style json
        # testset = loads_json(args.input_file, 'Load Test Set: '+args.input_file)
        # if args.subset > 0:
        #     logger.info("Decoding subset: %d", args.subset)
        #     testset = testset[:args.subset]
             
        with open(args.input_file, encoding="utf-8") as fin:
            data = json.load(fin)
        #     input_lines = [x.strip() for x in fin.readlines()]
        #     if args.subset > 0:
        #         logger.info("Decoding subset: %d", args.subset)
        #         input_lines = input_lines[:args.subset]
        # data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
        # input_lines = [data_tokenizer.tokenize(
        #     x)[:max_src_length] for x in input_lines]
        # input_lines = sorted(list(enumerate(input_lines)),
        #                      key=lambda x: -len(x[1]))
        # output_lines = [""] * len(input_lines)
        # score_trace_list = [None] * len(input_lines)
        # total_batch = math.ceil(len(input_lines) / args.batch_size)

        data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
        PQA_dict = {} #will store the generated distractors
        dis_tot = 0
        dis_n = 0
        len_tot = 0
        hypothesis = {}
        ##change to process one by one and store the distractors in PQA json form
        ##with tqdm(total=total_batch) as pbar:
        # for example in tqdm(testset):
        #     question_id = str(example['id']['file_id']) + '_' + str(example['id']['question_id'])
        #     if question_id in hypothesis:
        #         continue 
            # dis_n += 1
            # if dis_n % 2000 == 0:
            #     logger.info("Already processed: "+str(dis_n))
        counter = 0
        for race_id, example in tqdm(data.items()):
            counter += 1
            if args.subset > 0 and counter >= args.subset:
                break 
            eg_dict = {}
            # eg_dict["question_id"] = question_id
            # eg_dict["question"] = ' '.join(example['question'])
            # eg_dict["context"] = ' '.join(example['article'])
                
            eg_dict["question"] = example['question']
            eg_dict["context"] = example['context']
            label = int(example["label"])
            options = example["options"]
            answer = options[label]
            #new_distractors = []
            pred1 = None
            pred2 = None 
            pred3 = None
            #while next_i < len(input_lines):
            #_chunk = input_lines[next_i:next_i + args.batch_size]
            #line = example["context"].strip() + ' ' + example["question"].strip()
            question = example['question']
            question = question.replace('_', ' ')
            line = ' '.join(nltk.word_tokenize(example['context']) + nltk.word_tokenize(question))
            line = [data_tokenizer.tokenize(line)[:max_src_length]]
            # buf_id = [x[0] for x in _chunk]
            # buf = [x[1] for x in _chunk]
            buf = line 
            #next_i += args.batch_size
            max_a_len = max([len(x) for x in buf])
            instances = []
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline:
                    instances.append(proc(instance))
            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(
                    instances)
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                # for i in range(1):
                    #try max 10 times
                    # if len(new_distractors) >= 3:
                    #     break
                traces = model(input_ids, token_type_ids,
                               position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                    # print (np.array(output_ids).shape)
                    # print (output_ids)
                else:
                    output_ids = traces.tolist()
                # now only supports single batch decoding!!!
                # will keep the second and third sequence as backup
                for i in range(len(buf)):
                    # print (len(buf), buf)
                    for s in range(len(output_ids)):
                        output_seq = output_ids[s]
                        #w_ids = output_ids[i]
                        #output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_buf = tokenizer.convert_ids_to_tokens(output_seq)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        if s == 1:
                            backup_1 = output_tokens
                        if s == 2:
                            backup_2 = output_tokens 
                        if pred1 is None:
                            pred1 = output_tokens
                        elif jaccard_similarity(pred1, output_tokens) < 0.5:
                            if pred2 is None:
                                pred2 = output_tokens
                            elif pred3 is None:
                                if jaccard_similarity(pred2, output_tokens) < 0.5:
                                    pred3 = output_tokens
                        if pred1 is not None and pred2 is not None and pred3 is not None:
                            break
                    if pred2 is None:
                        pred2 = backup_1
                        if pred3 is None:
                            pred3 = backup_2
                    elif pred3 is None:
                        pred3 = backup_1 
                        # output_sequence = ' '.join(detokenize(output_tokens))
                        # print (output_sequence)
                        # print (output_sequence)
                        # if output_sequence.lower().strip() == answer.lower().strip():
                        #     continue 
                        # repeated = False
                        # for cand in new_distractors:
                        #     if output_sequence.lower().strip() == cand.lower().strip():
                        #         repeated = True
                        #         break 
                        # if not repeated:
                        #     new_distractors.append(output_sequence.strip())
            
            #hypothesis[question_id] = [pred1, pred2, pred3]
            new_distractors = [pred1, pred2, pred3]
            # print (new_distractors)
            # dis_tot += len(new_distractors)
            # # fill the missing ones with original distractors
            # for i in range(4):
            #     if len(new_distractors) >= 3:
            #         break 
            #     elif i == label:
            #         continue 
            #     else:
            #         new_distractors.append(options[i])
            for dis in new_distractors:
                len_tot += len(dis)
                dis_n += 1
            new_distractors = [' '.join(detokenize(dis)) for dis in new_distractors if dis is not None]
            assert len(new_distractors)==3, "Number of distractors WRONG"
            new_distractors.insert(label, answer)
            #eg_dict["generated_distractors"] = new_distractors 
            eg_dict["options"] = new_distractors
            eg_dict["label"] = label
            #PQA_dict[question_id] = eg_dict
            PQA_dict[race_id] = eg_dict

        # reference = {}
        # for example in testset:
        #     question_id = str(example['id']['file_id']) + '_' + str(example['id']['question_id'])
        #     if question_id not in reference.keys():
        #         reference[question_id] = [example['distractor']]
        #     else:
        #         reference[question_id].append(example['distractor'])
                 
        # _ = eval(hypothesis, reference)
        # assert len(PQA_dict) == len(data), "Number of examples WRONG"
        # logger.info("Average number of GENERATED distractor per question: "+str(dis_tot/dis_n))
        logger.info("Average length of distractors: "+str(len_tot/dis_n))
        with open(args.output_file, mode='w', encoding='utf-8') as f:
            json.dump(PQA_dict, f, indent=4)
                        #output_lines[buf_id[i]] = output_sequence
                        # if args.need_score_traces:
                        #     score_trace_list[buf_id[i]] = {
                        #         'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
                #pbar.update(1)
        # if args.output_file:
        #     fn_out = args.output_file
        # else:
        #     fn_out = model_recover_path+'.'+args.split
        # with open(fn_out, "w", encoding="utf-8") as fout:
        #     for l in output_lines:
        #         fout.write(l)
        #         fout.write("\n")

        # if args.need_score_traces:
        #     with open(fn_out + ".trace.pickle", "wb") as fout_trace:
        #         pickle.dump(
        #             {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
        #         for x in score_trace_list:
        #             pickle.dump(x, fout_trace)


if __name__ == "__main__":
    main()
