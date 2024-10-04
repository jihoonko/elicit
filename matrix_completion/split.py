import sys
import numpy as np
import random
import tqdm

random.seed(0)
np.random.seed(0)

# argv[1:] = [filename, ratio_train, ratio_val, ratio_test]

fname = sys.argv[1]
train, val, test = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
splits = [0 for _ in range(train)] + [1 for _ in range(val)] + [2 for _ in range(test)]    
    
with open(fname, 'r') as f:
    lines = f.readlines()
    lines = [line.replace('::', '\t').split() for line in tqdm.tqdm(lines)]
    item_map = {k: i+1 for i, k in enumerate(sorted(set([i for u, i, r, t in lines])))}
    lines = [(u, str(item_map[i]), r, t) for u, i, r, t in lines]
    print(len(item_map))
    
    targets = (np.random.permutation(len(lines)) % len(splits)).tolist()

fs = [open(fname[:-4] + '.train', 'w'), open(fname[:-4] + '.val', 'w'), open(fname[:-4] + '.test', 'w')]
for line, target in zip(lines, targets):
    fs[splits[target]].write('\t'.join(line) + '\n')

for f in fs: f.close()
