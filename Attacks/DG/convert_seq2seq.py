# convert to src-tgt pair files
import os
import glob
import tqdm
import json
import unicodedata
import logging
logger = logging.getLogger(__name__)


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False

def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False

def _clean_text(text):
  """Performs invalid character removal and whitespace cleanup on text."""
  output = []
  for char in text:
    cp = ord(char)
    if cp == 0 or cp == 0xfffd or _is_control(char):
      continue
    if _is_whitespace(char):
      output.append(" ")
    else:
      output.append(char)
  return "".join(output)


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
		logger.info("LOOKING AT {} train".format(data_dir))
		high = os.path.join(data_dir, "train/high")
		middle = os.path.join(data_dir, "train/middle")
		high = self._read_txt(high)
		middle = self._read_txt(middle)
		return self._create_examples(high + middle, "train")

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


def convert_seq2seq(examples, data_dir, set_type):
	#convert examples to src.txt and tgt.txt 
	expected = len(examples) * 3
	src_f = open(os.path.join(data_dir, "src-"+set_type+'.txt'), mode="w+")
	tgt_f = open(os.path.join(data_dir, "tgt-"+set_type+'.txt'), mode="w+")
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))
		label = int(example.label)

		passage = example.contexts[0]
		passage = _clean_text(passage.strip())
		question = example.question 
		question = _clean_text(question.strip())
		options = example.endings 
		enc = passage + ' ' + question

		for (idx, opt) in enumerate(options):
			if idx != label:
				opt = _clean_text(opt.strip())
				src_f.write(enc+'\n')
				tgt_f.write(opt+'\n')
	
	src_f.seek(0)
	tgt_f.seek(0)

	if len(src_f.readlines()) != expected:
		print ("Source file Line number WRONG!")
	else:
		print ("Source file loaded")

	if len(tgt_f.readlines()) != expected:
		print ("Target file Line number WRONG!")
	else:
		print ("Target file loaded")

	print ("Number of examples: ", expected)
	src_f.close()
	tgt_f.close()

if not os.path.exists('RACE_seq2seq'):
	os.mkdir('RACE_seq2seq')

processor = RaceProcessor()
test_examples = processor.get_test_examples('RACE')
convert_seq2seq(test_examples, 'RACE_seq2seq', 'test')

dev_examples = processor.get_dev_examples('RACE')
convert_seq2seq(dev_examples, 'RACE_seq2seq', 'dev')

train_examples = processor.get_train_examples('RACE')
convert_seq2seq(train_examples, 'RACE_seq2seq', 'train')






























