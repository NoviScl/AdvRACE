# convert QA files into PQA files that can be directly used for usage 
import os 
import glob
import tqdm
import json
import logging
import string

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
        print ("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        print ("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        print ("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
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



def convert_QA_to_PQA(
    examples,
    distractor_file,
    record_file):
    """
    convert nbest prediction file into top-3 distractors.

    distractor_file: the nbest_predictions file
    """

    PQA_dict = {}

    dis_n = 0 # total number of questions
    dis_tot = 0 # total number of generated distractors

    with open(distractor_file, "r", encoding="utf-8") as fin:
        dis_data = json.load(fin)

    print ('#Examples: ', len(examples))
    print ('#Dis Data: ', len(dis_data))

    # for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
    for race_id, example in tqdm.tqdm(examples.items()):
        # if ex_index % 5000 == 0:
        #     print ("Writing example %d of %d" % (ex_index, len(examples)))
        
        if race_id not in dis_data:
            print (race_id + ' not found in QA file. Skipped.')
            continue 

        eg = dis_data[race_id]
        options = eg["distractors"]
        # ans = example.endings[int(example.label)]
        ans = example['options'][int(example['label'])]
        options.insert(int(example['label']), ans)

        if len(options) != 4:
        	print ("Number of options is WRONG!")
        	print (options)
        	continue

        eg_dict = {}
        eg_dict["question"] = example['question']
        # eg_dict["context"] = example.contexts[0]
        eg_dict["context"] = example['context']
        eg_dict["options"] = options
        eg_dict["label"] = int(example['label'])
        
        # example_id = '/'.join(race_id.split('/')[-4:])
        # PQA_dict[example_id] = eg_dict 
        PQA_dict[race_id] = eg_dict
        

    with open(record_file, mode='w', encoding='utf-8') as f:
    	json.dump(PQA_dict, f, indent=4)

    if len(examples) == len(PQA_dict):
    	print ("Finish converting: ", distractor_file)
    
# construct para+charSwap+addSent+extractive:
with open('/ps2/intern/clsi/AddSent/test_dis_Para_CharSwapNoQn_AddSent.json', 'r') as f:
    data = json.load(f)
convert_QA_to_PQA(data, '/ps2/intern/clsi/RACE_extractive_distractors/test_dis.json', 'test_dis_Para_CharSwapNoQn_AddSent_Ext.json')

# dir_ = 'final_distractor_datasets'
# extractive_dir = os.path.join(dir_, 'extractive')
# gpt_dir = os.path.join(dir_, 'gpt2')

# if not os.path.exists(dir_):
#     os.mkdir(dir_)
# if not os.path.exists(extractive_dir):
#     os.mkdir(extractive_dir)
# if not os.path.exists(gpt_dir):
#     os.mkdir(gpt_dir)

# ext_qa_dir = 'RACE_extractive_distractors'
# gpt_qa_dir = 'gpt2-distractors'
# processor = RaceProcessor()

# # convert extractive
# test_examples = processor.get_test_examples('RACE')
# dev_examples = processor.get_dev_examples('RACE')
# train_examples = processor.get_train_examples('RACE')
# convert_QA_to_PQA(test_examples, distractor_file=os.path.join(ext_qa_dir, 'test_dis.json'), record_file=os.path.join(extractive_dir, 'test_dis.json'))
# convert_QA_to_PQA(dev_examples, distractor_file=os.path.join(ext_qa_dir, 'dev_dis.json'), record_file=os.path.join(extractive_dir, 'dev_dis.json'))
# convert_QA_to_PQA(train_examples, distractor_file=os.path.join(ext_qa_dir, 'train_dis.json'), record_file=os.path.join(extractive_dir, 'train_dis.json'))

# #convert gpt2
# test_examples = processor.get_test_examples('/ps2/intern/clsi/RACE')
# dev_examples = processor.get_dev_examples('/ps2/intern/clsi/RACE')
# train_examples = processor.get_train_examples('/ps2/intern/clsi/RACE')
# convert_QA_to_PQA(test_examples, distractor_file=os.path.join(gpt_qa_dir, 'test_dist.json'), record_file=os.path.join(gpt_dir, 'test_dis.json'))
# convert_QA_to_PQA(dev_examples, distractor_file=os.path.join(gpt_qa_dir, 'dev_dist.json'), record_file=os.path.join(gpt_dir, 'dev_dis.json'))
# convert_QA_to_PQA(train_examples, distractor_file=os.path.join(gpt_qa_dir, 'train_dist.json'), record_file=os.path.join(gpt_dir, 'train_dis.json'))







