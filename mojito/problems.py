import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support as prfs

from .utils import TextMod


class Problem:

    def set_fold(self, train_examples):
        raise NotImplementedError('virtual method')

    def e2u(self, X_explainable):
        raise NotImplementedError('virtual method')

    def evaluate(self, learner, X, Y):
        raise NotImplementedError('virtual method')

    def improve(self, example, y):
        raise NotImplementedError('virtual method')

    def improve_explanation(self, explainer, x_explainable, explanation):
        raise NotImplementedError('virtual method')
