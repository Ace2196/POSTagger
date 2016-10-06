from collections import defaultdict
import operator
import sys

# Populate program argument variables
FILE_SAMPLE = sys.argv[1]
FILE_OUTS = sys.argv[2]

sample_outs = []
outs = []

with open(FILE_SAMPLE, 'r') as f:
    for s in f:
        sample_outs.append(s.split())
with open(FILE_OUTS, 'r') as f:
    for s in f:
        outs.append(s.split())

correct = [pair for sent in sample_outs for pair in sent]
check = [pair for sent in outs for pair in sent]

mistake_count = defaultdict(int)

for i, check_pair in enumerate(check):
    correct_pair = correct[i]
    correct_tag = correct_pair.split('/')[-1]
    check_tag = check_pair.split('/')[-1]
    if correct_tag != check_tag:
        mistake_count['%s as %s'%(correct_tag, check_tag)] += 1
sorted_mistake_count = sorted(mistake_count.items(), key=operator.itemgetter(1), reverse=True)

print(sum(mistake_count.values()))
print(sorted_mistake_count[:10])
