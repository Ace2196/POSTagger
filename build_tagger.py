from __future__ import division
import itertools
from json import dump as json_dump
from collections import defaultdict
import sys

# Populate program argument variables
FILE_TRAIN = sys.argv[1]
FILE_DEVT = sys.argv[2]
FILE_OUT = sys.argv[3]

#Default variables
TAG_SENT_START = '<s>'
TAG_SENT_END = '<\s>'
UNKNOWN_WORD = '<UNK>'

sents = []
devt_sents = []
with open(FILE_TRAIN, 'r') as f:
    for s in f:
        sents.append(s.split())
with open(FILE_DEVT, 'r') as f:
    for s in f:
        devt_sents.append(s.split())

vocab = set()
tag_count = defaultdict(int)
word_tag_pair_count = defaultdict(int)
tag_tag_pair_count = defaultdict(int)

known_count = defaultdict(int)
unknown_count = defaultdict(int)

trans_prob = defaultdict(float)
obs_prob = defaultdict(float)

# Train on training set
for sent in sents:
    prev_tag = TAG_SENT_START
    tag_count[prev_tag] += 1
    for i, wordtag in enumerate(sent):
        tag = wordtag.split('/')[-1]
        word = '/'.join(wordtag.split('/')[:-1])
        # Update appropriate counts
        vocab.add(word)
        tag_count[tag] += 1
        tag_tag_pair_count['%s %s'%(prev_tag, tag)] += 1
        word_tag_pair_count[wordtag] += 1
        # Iteration step
        prev_tag = tag
    tag_count[TAG_SENT_END] += 1
    tag_tag_pair_count['%s %s'%(prev_tag, TAG_SENT_END)] += 1

# Tweak counts from training set with development set
vocab.add(UNKNOWN_WORD)
for tag in tag_count.keys():
    if tag != TAG_SENT_START and tag != TAG_SENT_END:
        for word in vocab:
            word_tag_pair_count.setdefault('%s/%s'%(word,tag), 0)
for sent in devt_sents:
    prev_tag = TAG_SENT_START
    tag_count[prev_tag] += 1
    for i, wordtag in enumerate(sent):
        tag = wordtag.split('/')[-1]
        word = '/'.join(wordtag.split('/')[:-1])
        # Update appropriate counts
        if word not in vocab:
            word = UNKNOWN_WORD
        tag_count[tag] += 1
        tag_tag_pair_count['%s %s'%(prev_tag, tag)] += 1
        word_tag_pair_count['%s/%s'%(word,tag)] += 1
        # Iteration step
        prev_tag = tag
    tag_count[TAG_SENT_END] += 1
    tag_tag_pair_count['%s %s'%(prev_tag, TAG_SENT_END)] += 1

# Calculation for smoothed transition probabilities
for tag1, tag2 in itertools.product(tag_count.keys(), tag_count.keys()):
    tag_tag_pair_count.setdefault('%s %s'%(tag1, tag2), 0)
for key in tag_tag_pair_count.keys():
    trans_prob[key] = (tag_tag_pair_count[key] + 1)/(tag_count[key.split()[0]] + len(tag_tag_pair_count.keys()))
tag_count.pop(TAG_SENT_START, None)
tag_count.pop(TAG_SENT_END, None)
# Calculation for emmission probabilities
for key in word_tag_pair_count.keys():
    obs_prob[key] = (word_tag_pair_count[key] + 1)/(tag_count[key.split('/')[-1]] + len(vocab))

with open(FILE_OUT, "w") as outfile:
    json_dump([list(tag_count.keys()), trans_prob, obs_prob], outfile, indent=4)
