import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support as prfs

from .utils import TextMod


class Problem:
    """A problem.

    Attributes
    ----------
    examples : list of int
        Indices of all examples in the dataset.
    X : ndarray of shape (n_examples, n_features)
        The input part of the examples.
    Y : ndarray of shape (n_examples,)
        The supervision.
    class_names: list of str
        The class names.

    Parameters
    ----------
    min_coeff : float, defaults to 1e-4
        Threshold on the LIME coefficients for computing the explanation perf.
    """
    def __init__(self, min_coeff=1e-4):
        self.min_coeff = min_coeff

    def wrap_preproc(self, model):
        raise NotImplementedError('virtual method')

    def explain(self, learner, known_examples, example,
                num_samples=None, num_features=None):
        raise NotImplementedError('virtual method')

    def improve(self, example, y):
        raise NotImplementedError('virtual method')

    def improve_explanation(self, explainer, x_explainable, explanation):
        raise NotImplementedError('virtual method')

    def get_explanation_perf(self, true_explanation, pred_explanation):
        raise NotImplementedError('virtual method')
