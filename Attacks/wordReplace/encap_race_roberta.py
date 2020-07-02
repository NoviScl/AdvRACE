import argparse
import glob
import logging
import os
import random

from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
	WEIGHTS_NAME,
	AdamW,
	BertConfig,
	BertForMultipleChoice,
	BertTokenizer,
	RobertaConfig,
	RobertaForMultipleChoice,
	RobertaTokenizer,
	get_linear_schedule_with_warmup,
)
from utils_multiple_choice import convert_examples_to_features, processors, InputExample, InputFeatures

processor = processors['new-race']()
label_list = processor.get_labels()
num_labels = len(label_list)
max_seq_length = 512

def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

## encap the roberta model into a class
class Model(nn.Module):
	def __init__(self, model_name_or_path):
		super(Model, self).__init__()
		self.config = RobertaConfig.from_pretrained(
			model_name_or_path,
			num_labels=num_labels,
		)
		self.tokenizer = RobertaTokenizer.from_pretrained(
			model_name_or_path,
			do_lower_case=False,
		)
		self.model = RobertaForMultipleChoice.from_pretrained(
			model_name_or_path,
			from_tf=False,
			config=self.config,
		)
		self.model = self.model.eval().cuda() 
		self.m = nn.Softmax(1)
		## output (loss, logits, etc)

	def forward(self, input_example):
		'''
		input: json dict 
		for now, process one example at a time 
		'''
		question = input_example['question']
		article = input_example['context'] 
		options = input_example['options'] 
		label = str(input_example['label'])

		examples = []
		examples.append(
			InputExample(
				example_id=None,
				question=question,
				contexts=[article, article, article, article],  # this is not efficient but convenient
				endings=[options[0], options[1], options[2], options[3]],
				label=label,
			)
		)

		features = convert_examples_to_features(
			examples,
			label_list,
			max_seq_length,
			self.tokenizer,
			pad_on_left=False,
			pad_token_segment_id=0)

		all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
		all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
		all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
		all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)


		with torch.no_grad():
			inputs = {
				"input_ids": all_input_ids.cuda(),
				"attention_mask": all_input_mask.cuda(),
				"labels": all_label_ids.cuda()
			}
			outputs = self.model(**inputs)
			tmp_eval_loss, logits = outputs[:2]
			logits = self.m(logits)

		return logits.cpu().numpy()
