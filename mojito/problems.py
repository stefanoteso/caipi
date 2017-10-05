import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support as prfs

from .utils import TextMod


class Problem:

    def e2u(self, X_explainable):
        raise NotImplementedError('virtual method')

    def evaluate(self, learner, X, Y):
        raise NotImplementedError('virtual method')

    def explain(self, coef, max_features=10):
        raise NotImplementedError('virtual method')

    def improve(self, example, y, g):
        raise NotImplementedError('virtual method')



class CancerProblem(Problem):
    """The breast cancer dataset.

    Features:
    - non explainable: 2nd degree homogeneous polynomial of the attributes
    - explainable: the attributes

    TODO: add support for multi-class classification.
    """
    def __init__(self, oracle, rng=None):
        from sklearn.datasets import load_breast_cancer

        dataset = load_breast_cancer()
        self.Y = dataset.target[dataset.target < 2]
        self.X_explainable = dataset.data[dataset.target < 2]
        self.X = self.e2u(self.X_explainable)
        self.examples = list(range(len(self.Y)))
        self.feature_names = dataset.feature_names

        oracle.fit(self.X, self.Y)
        self.oracle = oracle

    @staticmethod
    def _polynomial(a, b):
        x = np.array([ai*bj for ai in a for bj in b])
        return x / np.sum(np.abs(x))

    def e2u(self, X_explainable):
        if X_explainable.ndim == 1:
            return self._polynomial(X_explainable, X_explainable)
        return np.array([self._polynomial(x, x) for x in X_explainable])

    def evaluate(self, learner, X, Y):
        Y_hat = learner.predict(X)
        return prfs(Y, Y_hat, average='weighted')[:3]

    def explain(self, coef, max_features=10):
        indices = np.argsort(coef)[-max_features:]
        return coef[indices], indices

    def improve(self, y):
        return self.Y[example]

    @staticmethod
    def highlight(value):
        text = '{:5.3f}'.format(value)
        if np.abs(value) > 1e-6:
            return TextMod.BOLD + text + TextMod.END
        return text

#    def present_explanation(self, indices_weights, g):
#        indices, weights = g
#        names = self.feature_names[indices]
#        print('example {example} is though as {y} because'.format(**locals()))
#        print('\n'.join(['{:32s} = '.format(feature_name) + highlight(value)
#                          for name, weight in zip(names, weights)]))

    def improve_explanation(self, example, x_explainable, g):
        if g is None:
            return None
        explanation, indices = g
        true_explanation = explainer.explain(self, self.oracle, x_explainable)
        return (true_explanation + explanation) / 2, indices



class ReligionProblem(Problem):
    """Newsgroup problem over text documents."""
    def __init__(self, rng=None):
        from sklearn.datasets import fetch_20newsgroups

        CATEGORIES = ['alt.atheism', 'soc.religion.christian']
        dataset = fetch_20newsgroups(subset='all', categories=CATEGORIES,
                                     random_state=rng)
        self.Y = dataset.target
        self.examples = list(range(len(self.Y)))

        vectorizer = CountVectorizer(lowercase=False, ngram_range=(1, 2))
        self.vectorizer = vectorizer.fit(dataset.data)
        self.X = vectorizer.transform(dataset.data)

        vectorizer = TfidfVectorizer(lowercase=False)
        self.explainable_vectorizer = vectorizer.fit(dataset.data)
        self.X_explainable = vectorizer.transform(dataset.data)
