import pickle
import numpy as np
import scipy as sp


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)


def densify(x):
    try:
        x = x.toarray()
    except AttributeError:
        pass
    if x.shape[0] != 1:
        # if X[i] is already dense, densify(X[i]) is a no-op, so we get an x
        # of shape (n_features,) and we turn it into (1, n_features);
        # if X[i] is sparse, densify(X[i]) gives an x of shape (1, n_features).
        x = x[np.newaxis, ...]
    return x


def _stack(arrays, d_stack, s_stack):
    arrays = [a for a in arrays if a is not None]
    if len(arrays) == 0:
        return None
    if len(arrays) == 1:
        return arrays[0]
    if isinstance(arrays[0], sp.sparse.csr_matrix):
        return s_stack(arrays)
    else:
        return d_stack(arrays)

vstack = lambda arrays: _stack(arrays, np.vstack, sp.sparse.vstack)
hstack = lambda arrays: _stack(arrays, np.hstack, sp.sparse.hstack)


class PipeStep:
    def __init__(self, func):
        self.func = func

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return self.func(X)
