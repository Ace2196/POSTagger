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
UNKNOWN_WORD = '<NK>'

sents = []
devt_sents = []
with open(FILE_TRAIN, 'r') as f:
    for s in f:
        sents.append(s.split())
with open(FILE_DEVT, 'r') as f:
    for s in f:
        devt_sents.append(s.split())
sents_flat = list(itertools.chain(*sents))
devt_sents_flat = list(itertools.chain(*devt_sents))
word_tag_pair_count = defaultdict(int)
tag_count = defaultdict(int)
tag_tag_pair_count = defaultdict(int)

known_count = defaultdict(int)
unknown_count = defaultdict(int)

trans_prob = defaultdict(int)
obs_prob = defaultdict(int)
unk_prob = defaultdict(int)

for sent in sents:
    for i, word in enumerate(sent):
        tag = word.split('/')[-1]
        tag_count[tag] += 1
        word_tag_pair_count[word] += 1
        if i == 0:
            tag_count[TAG_SENT_START] += 1
            tag_tag_pair_count['%s %s'%(TAG_SENT_START, tag)] += 1
        elif i == (len(sent)-1):
            tag_count[TAG_SENT_END] += 1
            tag_tag_pair_count['%s %s'%(tag, TAG_SENT_END)] += 1
        else:
            next_tag = sents_flat[i+1].split('/')[-1]
            tag_tag_pair_count['%s %s'%(tag, next_tag)] += 1

# Calculation emission and smoothed transition probabilities
for tag1, tag2 in itertools.product(tag_count.keys(), tag_count.keys()):
    tag_tag_pair_count.setdefault('%s %s'%(tag1, tag2), 0)
for key in tag_tag_pair_count.keys():
    trans_prob[key] = (tag_tag_pair_count[key] + 1)/(tag_count[key.split()[0]] + len(tag_tag_pair_count.keys()))
for key in word_tag_pair_count.keys():
    obs_prob[key] = word_tag_pair_count[key]/tag_count[key.split('/')[-1]]
tag_count.pop(TAG_SENT_START, None)
tag_count.pop(TAG_SENT_END, None)

# Probability of unknown words given a tag, used to smooth emission probabilities
for word_tag_pair in devt_sents_flat:
    if word_tag_pair not in word_tag_pair_count:
        unknown_count[word_tag_pair.split('/')[-1]] += 1
for tag in tag_count.keys():
    unk_prob[tag] = unknown_count.get(tag,0)/len(devt_sents_flat)

with open(FILE_OUT, "w") as outfile:
    json_dump([list(tag_count.keys()), trans_prob, obs_prob, unk_prob], outfile, indent=4)
