'''
This is a processor that reads RACE files and turns it into Squad style.
We do not directly change the dataset itself, rather we use the data processor to do the answer-insertion.
'''

import numpy as np
import random
import nltk

random.seed(2020)

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class RaceExample(object):
	def __init__(
		self, 
		race_id,
		context_text,
		question_text,
		option_0,
		option_1,
		option_2,
		option_3,
		label=None
	):
		self.race_id = race_id
		self.context_text = context_text
		self.question_text = question_text
		self.options = [
			option_0,
			option_1,
			option_2,
			option_3,
		]
		self.label = label
		self.answer = self.options[label]

		## insert options into context
		## naive solution, random insert
		self.context_text = self.context_text.split()
		for opt in self.options: 
			self.context_text.insert(random.randrange(0, len(self.context_text)), opt)
		self.context_text = ' '.join(self.context_text)


       	## TODO: Locate answer span index 
       	self.start_position, self.end_position = 0, 0

       	doc_tokens = []
       	char_to_word_offset = []
       	prev_is_whitespace = True

       	# Split on whitespace so that different tokens may be attributed to their original position.
       	for c in self.context_text:
       		if _is_whitespace(c):




	def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_text}",
            f"question: {self.question_text}",
            f"option_0: {self.options[0]}",
            f"option_1: {self.options[1]}",
            f"option_2: {self.options[2]}",
            f"option_3: {self.options[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
	def __init__(
		self,
    	example_id,
    	choices_features,
    	label
    ):
    	self.example_id = example_id
    	self.choices_features = [
            {
                'tokens': tokens,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for tokens, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
       	

def read_race_examples(paths):
	## paths is a list containing all data folders
	examples = []
	for path in paths:
		filenames = glob.glob(path+"/*txt")
		for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id = filename+'-'+str(i),
                            context_text = article,
                            question_text = question,

                            option_0 = options[0],
                            option_1 = options[1],
                            option_2 = options[2],
                            option_3 = options[3],
                            label = truth))
                
    return examples 




































