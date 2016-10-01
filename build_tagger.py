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
    prev_tag = TAG_SENT_START
    tag_count[prev_tag] += 1
    for i, wordtag in enumerate(sent):
        tag = wordtag.split('/')[-1]
        tag_count[tag] += 1
        tag_tag_pair_count['%s %s'%(prev_tag, tag)] += 1
        word_tag_pair_count[wordtag] += 1
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
    obs_prob[key] = word_tag_pair_count[key]/tag_count[key.split('/')[-1]]

with open(FILE_OUT, "w") as outfile:
    json_dump([list(tag_count.keys()), trans_prob, obs_prob], outfile, indent=4)
