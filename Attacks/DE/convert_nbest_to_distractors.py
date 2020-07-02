# convert nbest files into filtered top-3 distractors 
import os 
import glob
import tqdm
import json
import logging
import string

# I've listed the nltk stopwords list here so that you don't need to download nltk corpus.
stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

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



def convert_nbest_to_distractors(
    examples,
    label_list,
    distractor_file,
    record_file):
    """
    convert nbest prediction file into top-3 distractors.

    distractor_file: the nbest_predictions file
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    QA_dict = {}

    dis_n = 0 # total number of questions
    dis_tot = 0 # total number of generated distractors

    with open(distractor_file, "r", encoding="utf-8") as fin:
        dis_data = json.load(fin)

    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            print ("Writing example %d of %d" % (ex_index, len(examples)))
        ans = example.endings[int(example.label)]
        orig_dis = [example.endings[i] for i in range(4) if i!=int(example.label)]
        ext_spans = []
        
        if example.example_id not in dis_data:
            print (example.example_id + ' not found in nbest file. Skipped.')
            continue 
        cands = dis_data[example.example_id]
        for cand in cands:
            if len(ext_spans) >= 3: #alrd have enough
                continue 
            if cand['text'] in string.punctuation:
                continue
            if cand['text'].lower() in stopwords:
                continue 
            if cand['text'].lower() in ans.lower():
                continue
            repeat = False
            for c in range(len(ext_spans)):
                if cand['text'].lower() in ext_spans[c].lower():
                    repeat = True
                    break
                elif ext_spans[c].lower() in cand['text'].lower():
                    #ext_spans[c] = cand['text']
                    repeat = True
                    break 
            if repeat:
                continue
            ext_spans.append(cand['text'])

        eg_dict = {}
        eg_dict["question"] = example.question
        eg_dict["answer"] = example.endings[int(example.label)]
        eg_dict["distractors"] = []

        dis_n += 1
        dis_tot += len(ext_spans)

        # for dis in orig_dis:
        # 	# if less than 3 available, use the original ones
        #     if len(ext_spans) > 0:
        #         ending = ext_spans[0]
        #         ext_spans.pop(0)
        #     eg_dict["distractors"].append(ending)

        eg_dict["distractors"] = ext_spans 
        i = 0
        while len(eg_dict["distractors"]) < 3:
            dis = orig_dis[i]
            repeat = False
            for d in eg_dict["distractors"]:
                if d.lower() == dis.lower():
                    repeat = True
                    break 
            if not repeat:
                eg_dict["distractors"].append(dis)
            i += 1


        label = label_map[example.label]
        QA_dict[example.example_id] = eg_dict 
        

    with open(record_file, mode='w', encoding='utf-8') as f:
    	json.dump(QA_dict, f, indent=4)
    print ("Number of orig distractors used: ", dis_n*3 - dis_tot)
    print ("Average number of generated distractors for each question: " + str(dis_tot/dis_n))



if not os.path.exists('RACE_extractive_distractors'):
    os.mkdir('RACE_extractive_distractors')
processor = RaceProcessor()
label_list = processor.get_labels()
test_examples = processor.get_test_examples('RACE')
convert_nbest_to_distractors(test_examples, label_list, distractor_file='output_albert_spanRACE_newV2_test_nbest/nbest_predictions_.json', record_file='RACE_extractive_distractors/test_dis.json')
train_examples = processor.get_train_examples('RACE')
convert_nbest_to_distractors(train_examples, label_list, distractor_file='output_albert_spanRACE_newV2_train_nbest/nbest_predictions_.json', record_file='RACE_extractive_distractors/train_dis.json')
dev_examples = processor.get_dev_examples('RACE')
convert_nbest_to_distractors(dev_examples, label_list, distractor_file='output_albert_spanRACE_newV2_dev_nbest/nbest_predictions_.json', record_file='RACE_extractive_distractors/dev_dis.json')







