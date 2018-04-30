#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from caipi import load

plt.style.use('ggplot')

data = load(sys.argv[2])

def nrm(w):
    return (w - np.min(w)) / (np.max(w) - np.min(w) + 1e-13)

try:
    data = np.array(data)
    # (n_folds, n_iters, n_classes, n_features)
    n_iters = data.shape[1]
    vectors = data[0,:,:,:].reshape((n_iters, -1))
except:
    vectors = np.vstack([data['w_train'], data['w_corr'], data['w_both']])

if vectors.shape[1] == 300:
    from numpy.linalg import lstsq

    RULE0_COORDS = {(0, 4), (0, 20), (0, 24), (4, 20), (4, 24), (20, 24)}
    RULE0_BASIS = np.array([1.0 if (i, j) in RULE0_COORDS else 0.0
                            for i in range(5*5)
                            for j in range(i+1, 5*5)])

    RULE1_COORDS = {(1, 2), (1, 3), (2, 3)}
    RULE1_BASIS = np.array([-1.0 if (i, j) in RULE1_COORDS else 0.0
                            for i in range(5*5)
                            for j in range(i+1, 5*5)])

    A = np.array([RULE0_BASIS, RULE1_BASIS]).T
    b = vectors.T

    alpha, residuals, _, _ = lstsq(A, b)
    # (n_rules, n_iters)

    fig, ax = plt.subplots(1, 1)
    x = np.arange(0, alpha.shape[1])
    coeff_rule0 = alpha[0,:]
    ax.plot(x, coeff_rule0, label='coeff. of rule 0', linewidth=2)
    coeff_rule1 = alpha[1,:]
    ax.plot(x, coeff_rule1, label='coeff. of rule 1', linewidth=2)
    ax.plot(x, residuals, label='residual', linewidth=2, linestyle=':')

    legend = ax.legend(loc='upper right',
                       shadow=False)

    fig.savefig(sys.argv[1] + '__coeff', bbox_inches='tight', pad_inches=0)
    quit()


fig = plt.figure(figsize=(50, len(vectors)))
ax = fig.add_subplot(111)
ax.set_axis_off()
ax.matshow(nrm(vectors), cmap=plt.get_cmap('gray'))
fig.savefig(sys.argv[1], bbox_inches='tight', pad_inches=0)
