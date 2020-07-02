from __future__ import print_function
from keras.utils import np_utils
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import keras
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import pickle

import numpy as np
np.random.seed(1337)  # for reproducibility

## load RACE data 
with open('/ps2/intern/clsi/RACE/test.json', 'r') as f:
    test_data = json.load(f)
# with open('/ps2/intern/clsi/RACE/train.json', 'r') as f:
#     test_data = json.load(f)

## preprocess the contexts
test_text = [eg["context"] for k,eg in test_data.items()]
# train_text = [eg["context"] for eg in train_data]

## smaller vocab for test set (15k)
VOCAB = 20000
tokenizer = Tokenizer(num_words=VOCAB, lower=False, filters='')
tokenizer.fit_on_texts(test_text)

## Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
# VOCAB = len(tokenizer.word_counts) + 1
# LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

# def to_seq(X): return pad_sequences(
#     tokenizer.texts_to_sequences(X), maxlen=100, padding='post')

## clsi: I didn't add padding here
def to_seq(X): 
    return tokenizer.texts_to_sequences(X)

# def prepare_data(data): return (to_seq(data[0]), to_seq(data[1]), data[2])

if __name__ == '__main__':
    test_context = to_seq(test_text)

    with open('cache/race_test_tokenizer_v20k.pkl', 'wb') as fh:
        pickle.dump(tokenizer, fh)
    with open('cache/tokenized_race_test_context.pkl', 'wb') as fh:
        pickle.dump(test_context, fh)
    