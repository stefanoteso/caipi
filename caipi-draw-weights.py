#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from caipi import load

data = load(sys.argv[2])

def nrm(w):
    return (w - np.min(w)) / (np.max(w) - np.min(w) + 1e-13)

try:
    data = np.array(data)
    # (n_folds, n_iters, n_classes, n_features)
    n_iters = data.shape[1]
    vectors = data[0,:,:,:].reshape((n_iters, -1))
except:
    vectors = [data['w_train'], data['w_corr'], data['w_both']]

print(vectors.shape)

fig = plt.figure(figsize=(50, len(vectors)))
ax = fig.add_subplot(111)
ax.matshow(nrm(vectors), cmap=plt.get_cmap('gray'))
#for i, w in enumerate(vectors):
#    ax = fig.add_subplot(len(vectors), 1, i + 1)
#    ax.matshow(nrm(w), cmap=plt.get_cmap('gray'))
fig.savefig(sys.argv[1], bbox_inches='tight', pad_inches=0)
