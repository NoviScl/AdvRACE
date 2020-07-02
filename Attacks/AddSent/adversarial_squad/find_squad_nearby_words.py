"""Find nearby words for words in SQuAD questions."""
"""Codes are modified to be compatible with Python3"""
import argparse
import json
from nltk.tokenize import word_tokenize
import numpy as np
from scipy import spatial
from sklearn.neighbors import KDTree
import string
import sys
from tqdm import tqdm

OPTS = None
RACE_DEV_FILE = '/ps2/intern/clsi/RACE/dev.json'
RACE_TEST_FILE = '/ps2/intern/clsi/RACE/test.json'
PUNCTUATION = set(string.punctuation) | set(['``', "''"])

def parse_args():
  parser = argparse.ArgumentParser('Find nearby words for words in SQuAD questions.')
  parser.add_argument('wordvec_file', help='File with word vectors.')
  parser.add_argument('--race-file', '-f',
                      help=('RACE file (defaults to RACE TEST file).'),
                      default=RACE_TEST_FILE)
  parser.add_argument('--output-file', '-o',
                      help=('DIR to store output json file'),
                      default='out/nearby_n100_glove_6B_100d.json')
  parser.add_argument('--num-neighbors', '-n', type=int, default=1,
                      help='Number of neighbors per word (default = 1).')
  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit(1)
  return parser.parse_args()

# extract all question and option words in RACE
def extract_words():
  with open(OPTS.race_file) as f:
    dataset = json.load(f)
  words = set()
  for race_id, example in dataset.items():
    cur_words = set(w.lower() for w in word_tokenize(example["question"]) if w not in PUNCTUATION)
    words |= cur_words
    for opt in example["options"]:
      cur_words = set(w.lower() for w in word_tokenize(opt) if w not in PUNCTUATION)
      words |= cur_words
  return words

def get_nearby_words(main_words):
  main_inds = {}
  all_words = []
  all_vecs = []
  with open(OPTS.wordvec_file) as f:
    for i, line in tqdm(enumerate(f)):
      toks = line.rstrip().split(' ')
      word = str(toks[0]).strip()
      vec = np.array([float(x) for x in toks[1:]])
      all_words.append(word)
      all_vecs.append(vec)
      if word in main_words:
        main_inds[word] = i
  print ('Found vectors for %d/%d words = %.2f%%' % (
      len(main_inds), len(main_words), 100.0 * len(main_inds) / len(main_words)), file=sys.stderr)
  all_vecs = np.array(all_vecs)
  tree = KDTree(all_vecs)
  nearby_words = {}
  for word in tqdm(main_inds):
    dists, inds = tree.query([all_vecs[main_inds[word]]],
                             k=OPTS.num_neighbors + 1)
    nearby_words[word] = [{'word': all_words[i], 'dist': d} for d, i in zip(dists[0], inds[0])]
  return nearby_words

def main():
  words = extract_words()
  print ('Found %d words' % len(words), file=sys.stderr)
  nearby_words = get_nearby_words(words)
  with open(OPTS.output_file, 'w') as f:
    json.dump(nearby_words, f, indent=4)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

