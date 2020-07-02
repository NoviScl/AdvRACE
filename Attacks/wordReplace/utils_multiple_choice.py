# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import logging
import os
import string 
from typing import List
import nltk
import string 
import random 
random.seed(2020)

import tqdm

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        examples = self._create_examples(high + middle, "train")
        random.shuffle(examples)
        return examples 

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        #return self._create_examples(high, "test")
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file #RACE/test/high/1001.txt
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            # race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                race_id = data_raw["race_id"] + '-' + str(i)
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples



class NewRaceProcessor(DataProcessor):
    """
    Processor for the RACE data set.
    Specifically for our own created json files with generated distractors.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        train_dir = os.path.join(data_dir, "train_dis.json")
        train = self._read_txt(train_dir)
        return self._create_examples(train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        dev_dir = os.path.join(data_dir, "dev_dis.json")
        dev = self._read_txt(dev_dir)
        return self._create_examples(dev, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        test_dir = os.path.join(data_dir, "test_dis.json")
        test = self._read_txt(test_dir)
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        with open(input_dir, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data 

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for race_id, example in lines.items():
            question = example["question"]
            article = example["context"]
            options = example["options"]
            truth = str(example["label"])            

            examples.append(
                InputExample(
                    example_id=race_id,
                    question=question,
                    contexts=[article, article, article, article],  # this is not efficient but convenient
                    endings=[options[0], options[1], options[2], options[3]],
                    label=truth,
                )
            )
        logger.info("# New Race Examples: "+ str(len(examples)))
        return examples


class C3Processor(DataProcessor):
    """
    Processor for the C3 data set.
    Specifically for our own created json files with generated distractors.
    """

    def get_train_m_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train-m".format(data_dir))
        train_dir = os.path.join(data_dir, "c3-m-train.json")
        train = self._read_txt(train_dir)
        return self._create_examples(train, "train")

    def get_train_d_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train-d".format(data_dir))
        train_dir = os.path.join(data_dir, "c3-d-train.json")
        train = self._read_txt(train_dir)
        return self._create_examples(train, "train")

    def get_train_examples(self, data_dir):
        return self.get_train_m_examples(data_dir) + self.get_train_d_examples(data_dir)

    def get_dev_m_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev-m".format(data_dir))
        dev_dir = os.path.join(data_dir, "c3-m-dev.json")
        dev = self._read_txt(dev_dir)
        return self._create_examples(dev, "dev")

    def get_dev_d_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev-d".format(data_dir))
        dev_dir = os.path.join(data_dir, "c3-d-dev.json")
        dev = self._read_txt(dev_dir)
        return self._create_examples(dev, "dev")

    def get_dev_examples(self, data_dir):
        return self.get_dev_m_examples(data_dir) + self.get_dev_d_examples(data_dir)

    def get_test_m_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test-m".format(data_dir))
        test_dir = os.path.join(data_dir, "c3-m-test.json")
        test = self._read_txt(test_dir)
        return self._create_examples(test, "test")

    def get_test_examples(self, data_dir):
        return self.get_test_m_examples(data_dir) + self.get_test_d_examples(data_dir)

    def get_test_d_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test-d".format(data_dir))
        test_dir = os.path.join(data_dir, "c3-d-test.json")
        test = self._read_txt(test_dir)
        return self._create_examples(test, "test")

    def get_labels(self):
        """Max 4 options per question. Some may have only 2 or 3 options."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        with open(input_dir, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data 

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets. Each question refers to one example."""
        examples = []
        for lst in lines:
            context = ' '.join(lst[0])
            c_id = lst[-1]
            index = 0
            for qa in lst[1]:
                question = qa["question"]
                options = qa["choice"]
                # fill in empty options for easier compute
                while len(options)<4:
                    options.append("")
                answer = qa["answer"]
                truth = str(options.index(answer))

                examples.append(
                    InputExample(
                        example_id=c_id + '-' + str(index),
                        question=question,
                        contexts=[context]*4,  # this is not efficient but convenient
                        endings=options,
                        label=truth,
                    )
                )

                index += 1
        logger.info("# C3 Examples: "+ str(len(examples)))
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


abbreviations = {'dr.': 'doctor', 'mr.': 'mister', 'bro.': 'brother', 'bro': 'brother', 'mrs.': 'mistress', 'ms.': 'miss', 'jr.': 'junior', 'sr.': 'senior',
                 'i.e.': 'for example', 'e.g.': 'for example', 'vs.': 'versus'}
terminators = ['.', '!', '?']
wrappers = ['"', "'", ')', ']', '}']


def find_sentences(paragraph):
   end = True
   sentences = []
   while end > -1:
       end = find_sentence_end(paragraph)
       if end > -1:
           sentences.append(paragraph[end:].strip())
           paragraph = paragraph[:end]
   sentences.append(paragraph)
   sentences.reverse()
   return sentences


def find_sentence_end(paragraph):
    [possible_endings, contraction_locations] = [[], []]
    contractions = abbreviations.keys()
    sentence_terminators = terminators + [terminator + wrapper for wrapper in wrappers for terminator in terminators]
    for sentence_terminator in sentence_terminators:
        t_indices = list(find_all(paragraph, sentence_terminator))
        possible_endings.extend(([] if not len(t_indices) else [[i, len(sentence_terminator)] for i in t_indices]))
    for contraction in contractions:
        c_indices = list(find_all(paragraph, contraction))
        contraction_locations.extend(([] if not len(c_indices) else [i + len(contraction) for i in c_indices]))
    possible_endings = [pe for pe in possible_endings if pe[0] + pe[1] not in contraction_locations]
    if len(paragraph) in [pe[0] + pe[1] for pe in possible_endings]:
        max_end_start = max([pe[0] for pe in possible_endings])
        possible_endings = [pe for pe in possible_endings if pe[0] != max_end_start]
    possible_endings = [pe[0] + pe[1] for pe in possible_endings if sum(pe) > len(paragraph) or (sum(pe) < len(paragraph) and paragraph[sum(pe)] == ' ')]
    end = (-1 if not len(possible_endings) else max(possible_endings))
    return end


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


def charSwap(text):
    if len(text)<4:
        return text 
    if text.lower() in stopwords:
        return text
    r = random.randint(1, len(text)-3)
    if text[r] in string.punctuation or text[r+1] in string.punctuation:
        return text
    text = list(text)
    text[r], text[r+1] = text[r+1], text[r]
    text = ''.join(text)
    return text

        
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    distractor_mode=None,
    distractor_file=None,
    record_file=None,
    attack_mode=None 
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`

    distractor_mode: 
    random: random select spans with same length as the answer
    extract: use n_best predictions 

    if use extractive mode, need to provide the path to the n_best predictions file,
    e.g. nbest_predictions_.json 

    attack_mode: addsent2pas-shuffle, addsent2opt-shuffle, addans2opt-shuffle

    record_file: json file to save the (passage), question, options
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    # logger.info("Distractor mode: "+str(distractor_mode))
    # logger.info("Attack mode: "+str(attack_mode))
    # attack_modes = ['addsent2pas-shuffle', 'addsent2opt-shuffle', 'addans2opt-shuffle', 'addsent2opt', 'addans2opt']
    # if attack_mode is not None and attack_mode.lower() not in attack_modes:
    #     logger.info("WRONG attack mode!")

    total_len = 0
    total_i = 0
    features = []
    PQA_dict = {}
    
    dis_n = 0 # total number of questions
    dis_tot = 0 # total number of generated distractors
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        ans = example.endings[int(example.label)]
        ans_len = len(ans.split())
        rand_spans = [] # store already selected random distractor spans' start pos
        ext_spans = [] # extracted spans after filtering 
        # select distractors
        

        eg_dict = {}
        label = int(example.label)
        eg_dict['question'] = example.question
        eg_dict['answer'] = example.endings[int(example.label)]
        eg_dict['distractors'] =[]
        answer = example.endings[int(example.label)]
        question = example.question
        distractors = [example.endings[i] for i in range(len(example.endings)) if i!=label]

        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            
            if ending_idx != int(example.label):
                total_len += len(ending.split())
                total_i += 1

            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]
        PQA_dict[example.example_id] = eg_dict
        dis_n += 1
        dis_tot += len(eg_dict['distractors'])

        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("race_id: {}".format(example.example_id))
        #     for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
        #         logger.info("choice: {}".format(choice_idx))
        #         logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
        #         logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
        #         logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
        #         logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))
    # if record_file:
    #     with open(record_file, mode='w', encoding='utf-8') as f:
    #         json.dump(PQA_dict, f, indent=4)
    #     logger.info('Average number of generated distractors for each question: '+str(dis_tot/dis_n))
        
    # logger.info('Average distractor len: '+str(total_len/total_i))
    return features


processors = {"race": RaceProcessor, "swag": SwagProcessor, "arc": ArcProcessor, "new-race": NewRaceProcessor, "c3": C3Processor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race": 4, "swag": 4, "arc": 4, "new-race": 4}
