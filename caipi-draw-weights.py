#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from caipi import load


def nrm(w):
    return (w - np.min(w)) / (np.max(w) - np.min(w) + 1e-13)


def nstd(x):
    return np.std(x, axis=0) / np.sqrt(x.shape[0])


plt.style.use('ggplot')

data = load(sys.argv[2])
data = np.array(data)
n_folds, n_iters, n_classes, n_features = data.shape

if n_features == 300:

    RULE0_COORDS = {(0, 4), (0, 20), (0, 24), (4, 20), (4, 24), (20, 24)}
    RULE0_BASIS = np.array([1.0 if (i, j) in RULE0_COORDS else 0.0
                            for i in range(5*5)
                            for j in range(i+1, 5*5)])

    RULE1_COORDS = {(1, 2), (1, 3), (2, 3)}
    RULE1_BASIS = np.array([-1.0 if (i, j) in RULE1_COORDS else 0.0
                            for i in range(5*5)
                            for j in range(i+1, 5*5)])

    DICTIONARY = np.array([RULE0_BASIS, RULE1_BASIS]).T

    results = []
    for k in range(n_folds):
        weights = data[k, :, 0, :]
        # (n_iters, n_features)
        alpha, residuals, _, _ = np.linalg.lstsq(DICTIONARY, weights.T,
                                                 rcond=None)
        # (n_rules, n_iters), (n_iters,)
        result = np.vstack((alpha, residuals.T))
        # (n_rules + 1, n_iters)
        results.append(result)

    results = np.array(results)
    # (n_folds, n_rules + 1, n_iters)

    def plot_both(ax, results, what, label, linestyle='-'):
        temp = results[:, what, :].reshape((n_folds, -1))
        x = np.arange(temp.shape[-1])
        y, yerr = np.mean(temp, axis=0), nstd(temp)
        ax.plot(x, y, linewidth=2, label=label, linestyle=linestyle)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.35, linewidth=0)

    fig, ax = plt.subplots(1, 1)
    plot_both(ax, results, 0, 'coeff. rule 0')
    plot_both(ax, results, 1, 'coeff. rule 1')
    plot_both(ax, results, 2, 'residual', linestyle=':')

    legend = ax.legend(loc='upper right',
                       shadow=False)

    fig.savefig(sys.argv[1] + '__coeff', bbox_inches='tight', pad_inches=0)

fig = plt.figure(figsize=(30, 100))
ax = fig.add_subplot(111)
ax.set_axis_off()

matrix = data.reshape((n_folds, n_iters, n_classes * n_features))
matrix = matrix.mean(axis=0)

ax.matshow(nrm(matrix), cmap=plt.get_cmap('gray'))

fig.savefig(sys.argv[1] + '__weights', bbox_inches='tight', pad_inches=0)
