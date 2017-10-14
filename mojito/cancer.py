import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support as prfs

from .problems import Problem
from .utils import TextMod


class CancerProblem(Problem):
    """The breast cancer dataset.

    Features:
    - non explainable: 2nd degree homogeneous polynomial of the attributes
    - explainable: the attributes

    TODO: add support for multi-class classification.
    """
    def __init__(self, oracle=None, rng=None):
        self.oracle = oracle

        dataset = load_breast_cancer()
        self.Y = dataset.target[dataset.target < 2]
        self.X_explainable_ = dataset.data[dataset.target < 2].astype(np.float32)
        self.X_ = self.e2u(self.X_explainable_).astype(np.float32)
        self.examples = list(range(len(self.Y)))
        self.feature_names = dataset.feature_names

    def set_fold(self, train_examples):
        self.scaler = MinMaxScaler().fit(self.X_[train_examples])
        self.X = self.scaler.transform(self.X_)

        scaler = MinMaxScaler().fit(self.X_explainable_[train_examples])
        self.X_explainable = scaler.transform(self.X_explainable_)

        if self.oracle:
            self.oracle.fit(self.X, self.Y)

    @staticmethod
    def polynomial_(a, b):
        return np.array([ai*bj for ai in a for bj in b])

    def e2u(self, X_explainable):
        if X_explainable.ndim == 1:
            return self.polynomial_(X_explainable, X_explainable)
        X = np.array([self.polynomial_(x, x) for x in X_explainable])
        return self.scaler.transform(X) if hasattr(self, 'scaler') else X

    def evaluate(self, learner, X, Y):
        Y_hat = learner.predict(X)
        return prfs(Y, Y_hat, average='weighted')[:3]

    def improve(self, example, y):
        return self.Y[example]

    @staticmethod
    def highlight(value):
        text = '{:+2.0f}'.format(value)
        return TextMod.BOLD + \
               (TextMod.RED if value < 0 else TextMod.BLUE) + \
               text + \
               TextMod.END

    def interact_(self, x_explainable, y, explanation):
        label = {
            0: TextMod.RED + 'negative',
            1: TextMod.GREEN + 'positive'
        }[y]
        print('the computer thinks that this example is {} because'.format(
                TextMod.BOLD + label + TextMod.END))

        indices = np.flatnonzero(explanation)
        polarity = explanation[indices]
        names = self.feature_names[indices]
        print('\n'.join(['{:32s} = '.format(name) + self.highlight(value)
                          for name, value in zip(names, polarity)]))

        # TODO: read off the user's explanation

    def improve_explanation(self, explainer, x_explainable, y, explanation):
        if explanation is None:
            return None, None, None, -1, None, None
        if self.oracle:
            return explainer.explain(self, self.oracle, x_explainable)
        else:
            return self.interact_(x_explainable, y, explanation)
