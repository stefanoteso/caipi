import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
    X_lime : ndarray of shape (n_examples, n_interpretable_features)
        Same as the above, but over the interpretable features.
    Y : ndarray of shape (n_examples,)
        The supervision.
    """
    def get_class_name(self, y):
        """Returns the colorized class name for the given label."""
        class_color = TextMod.BOLD + TextMod.GREEN if y else TextMod.RED
        return class_color + self.class_names[y] + TextMod.END

    def explain(self, learner, known_examples, example,
                num_samples=None, num_features=None):
        raise NotImplementedError('virtual method')

    def improve(self, example, y):
        raise NotImplementedError('virtual method')

    def improve_explanation(self, explainer, x_explainable, explanation):
        raise NotImplementedError('virtual method')
