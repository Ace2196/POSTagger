from __future__ import division
import numpy as np
from json import load as json_load
import sys

# Populate program argument variables
FILE_TEST = sys.argv[1]
FILE_MODEL = sys.argv[2]
FILE_SAMPLE = sys.argv[3]
FILE_OUT = sys.argv[4]

#Default variables
TAG_SENT_START = '<s>'
TAG_SENT_END = '<\s>'
UNKNOWN_WORD = '<UNK>'

known_correct = 0
known_total = 0
unknown_correct = 0
unknown_total = 0

sents = []
sample_outs = []
with open(FILE_TEST, 'r') as f:
    for s in f:
        sents.append(['/'.join(tok.split('/')[:-1]) for tok in s.split()])
        # sents.append(s.split())
        if len(sents) == 100:
            break
with open(FILE_SAMPLE, 'r') as f:
    for s in f:
        sample_outs.append(s)
        if len(sample_outs) == 100:
            break
with open(FILE_MODEL) as model_file:
    model = json_load(model_file)
fo = open(FILE_OUT,'w')
tags = model[0]
N = len(tags)
trans_prob = model[1]
obs_prob = model[2]

def get_trans_prob(t1,t2):
    return trans_prob['%s %s'%(t1,t2)]

def get_obs_prob(w,t):
    return obs_prob.get('%s/%s'%(w,t),
                        obs_prob['%s/%s'%(UNKNOWN_WORD,t)]) #obs_prob['%s/%s'%(UNKNOWN_WORD,t)]

for sn,sent in enumerate(sents):
    viterbi = np.zeros((N, len(sent)))
    backpt = np.ones((N, len(sent)), 'int32') * -1
    # Initialization step
    for i,tag in enumerate(tags):
        viterbi[i,0] = np.log(get_trans_prob(TAG_SENT_START,tag)) +\
                        np.log(get_obs_prob(sent[0],tag))
        backpt[i,0] = 0

    for t in range(1, len(sent)):
        for i,tag in enumerate(tags):
            # START of Maximization block
            val_max = -np.inf
            arg_max = -1
            for j,prev_tag in enumerate(tags):
                val_this = viterbi[j,t-1] + np.log(get_trans_prob(prev_tag,tag))
                if val_this >= val_max:
                    val_max = val_this
                    arg_max = j
            # END of Maximization block
            if get_obs_prob(sent[t],tag) == 0:
                if obs_prob['%s/%s'%(UNKNOWN_WORD,tag)] == 0:
                    print('%s/%s'%(sent[t],tag))
            viterbi[i,t] = val_max + np.log(get_obs_prob(sent[t],tag))
            backpt[i,t] = arg_max

    # START of Maximization block
    val_max = -np.inf
    arg_max = -1
    for j,prev_tag in enumerate(tags):
        val_this = viterbi[j,len(sent)-1] + np.log(get_trans_prob(prev_tag,TAG_SENT_END))
        if val_this >= val_max:
            val_max = val_this
            arg_max = j
    # END of Maximization block
    expected_tags = sample_outs[sn].strip().split(' ')
    sent_tags = ['']*len(sent)
    # sent_tags[-1] = tags[arg_max]
    for t in range(len(sent)-1, -1, -1):
        sent_tags[t] = tags[arg_max]
        if expected_tags[t].split('/')[-1] == sent_tags[t]:
            known_correct += 1
        arg_max = backpt[arg_max,t]
    sent_tagged = ['%s/%s'%(word,sent_tags[i]) for i,word in enumerate(sent)]
    fo.write(' '.join(sent_tagged)+'\n')
    # print(' '.join(sent_tagged))
    # print(sample_outs[sn])
    known_total += len(expected_tags)
    print(known_correct/known_total)
fo.close()
