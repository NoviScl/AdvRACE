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
    record_file):
    """
    convert nbest prediction file into top-3 distractors.

    distractor_file: the nbest_predictions file
    """

    PQA_dict = {}

    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            print ("Writing example %d of %d" % (ex_index, len(examples)))
    
        eg_dict = {}
        eg_dict["question"] = example.question
        eg_dict["context"] = example.contexts[0]
        eg_dict["options"] = example.endings
        eg_dict["label"] = int(example.label)
        
        example_id = '/'.join(example.example_id.split('/')[-4:])
        PQA_dict[example_id] = eg_dict 
        

    with open(record_file, mode='w', encoding='utf-8') as f:
    	json.dump(PQA_dict, f, indent=4)

    if len(examples) == len(PQA_dict):
    	print ("Finish converting and number is correct: ", len(examples))
    else:
        print ("Something wrong!")
    

dir = "/ps2/intern/clsi/RACE"
processor = RaceProcessor()

# convert extractive
test_examples = processor.get_test_examples('RACE')
dev_examples = processor.get_dev_examples('RACE')
train_examples = processor.get_train_examples('RACE')
convert_QA_to_PQA(test_examples, record_file=os.path.join(dir, 'test.json'))
convert_QA_to_PQA(dev_examples, record_file=os.path.join(dir, 'dev.json'))
convert_QA_to_PQA(train_examples, record_file=os.path.join(dir, 'train.json'))





