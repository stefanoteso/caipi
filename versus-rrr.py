#!/usr/bin/env python3

import sys

sys.path.append('../rrr')

import numpy as np
import matplotlib.pyplot as plt
from multilayer_perceptron import MultilayerPerceptron
from figure_grid import *
from local_linear_explanation import explanation_grid
import decoy_mnist


np.random.seed(0)

NUM_EPOCHS = 64 # this is the default in rrr


def correct_one(x, e, label, n_counterexamples):
    x = x.reshape((28, 28)) # 28 x 28 x [0 ... 255]
    e = e.reshape((28, 28)) # 28 x 28 x {False, True}

    X_counterexamples = []
    for _ in range(n_counterexamples):
        x_counterexample = np.array(x, copy=True)
        x_counterexample[e] = np.random.randint(0, 256, size=x[e].shape)
        X_counterexamples.append(x_counterexample.ravel())
    return X_counterexamples


def get_corrections(X, E, y, n_counterexamples=2):
    X_counterexamples, y_counterexamples = [], []
    for x, e, label in zip(X, E, y):
        temp = correct_one(x, e, label, n_counterexamples)
        X_counterexamples.extend(temp)
        y_counterexamples.extend([label] * len(temp))
    return np.array(X_counterexamples), np.array(y_counterexamples)


(X_tr_orig, X_tr_decoy, y_tr, E_tr,
 X_ts_orig, X_ts_decoy, y_ts, E_ts) = \
    decoy_mnist.generate_dataset(cachefile='../data/fashion/decoy-fashion.npz')

for n_counterexamples in range(1, 6):
    print('Fitting MLP corrected on {} examples'.format(n_counterexamples))

    X_tr_corrections, y_tr_corrections = get_corrections(X_tr_decoy, E_tr, y_tr,
                                                         n_counterexamples=n_counterexamples)
    X_tr_corrected = np.vstack([X_tr_decoy, X_tr_corrections])
    y_tr_corrected = np.hstack([y_tr, y_tr_corrections])

    print('# examples =', len(y_tr_corrected))

    mlp_corrected = MultilayerPerceptron()
    mlp_corrected.fit(X_tr_corrected, y_tr_corrected, num_epochs=NUM_EPOCHS,
                      verbose=100)
    print('avg. acc. on train (decoy) ', mlp_corrected.score(X_tr_decoy, y_tr))
    print('avg. acc. on test (decoy)  ', mlp_corrected.score(X_ts_decoy, y_ts))
    print('avg. acc. on test (nodecoy)', mlp_corrected.score(X_ts_orig, y_ts))

quit()

print('Fitting MLP annotated')
mlp_annotated = MultilayerPerceptron(l2_grads=1000)
mlp_annotated.fit(X_tr_decoy, y_tr, E_tr, num_epochs=NUM_EPOCHS)
print('avg. acc. on train (decoy) ', mlp_annotated.score(X_tr_decoy, y_tr))
print('avg. acc. on test (decoy)  ', mlp_annotated.score(X_ts_decoy, y_ts))
print('avg. acc. on test (nodecoy)', mlp_annotated.score(X_ts_orig, y_ts))

print('Fitting MLP normal')
mlp_normal = MultilayerPerceptron()
mlp_normal.fit(X_tr_decoy, y_tr, num_epochs=NUM_EPOCHS)
print('avg. acc. on train (decoy) ', mlp_normal.score(X_tr_decoy, y_tr))
print('avg. acc. on test (decoy)  ', mlp_normal.score(X_ts_decoy, y_ts))
print('avg. acc. on test (nodecoy)', mlp_normal.score(X_ts_orig, y_ts))

print('Fitting MLP nodecoy')
mlp_nodecoy = MultilayerPerceptron()
mlp_nodecoy.fit(X_tr_orig, y_tr, num_epochs=NUM_EPOCHS)
print('avg. acc. on train (decoy) ', mlp_nodecoy.score(X_tr_decoy, y_tr))
print('avg. acc. on test (decoy)  ', mlp_nodecoy.score(X_ts_decoy, y_ts))
print('avg. acc. on test (nodecoy)', mlp_nodecoy.score(X_ts_orig, y_ts))
