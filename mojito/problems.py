import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support as prfs

from .utils import TextMod


class Problem:

    def e2u(self, X_explainable):
        raise NotImplementedError('virtual method')

    def evaluate(self, learner, X, Y):
        raise NotImplementedError('virtual method')

    def improve(self, example, y):
        raise NotImplementedError('virtual method')

    def improve_explanation(self, explainer, x_explainable, explanation):
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

        self.oracle.fit(self.X, self.Y)

    @staticmethod
    def _polynomial(a, b):
        return np.array([ai*bj for ai in a for bj in b])

    def e2u(self, X_explainable):
        if X_explainable.ndim == 1:
            return self._polynomial(X_explainable, X_explainable)
        X = np.array([self._polynomial(x, x) for x in X_explainable])
        return self.scaler.transform(X) if hasattr(self, 'scaler') else X

    def evaluate(self, learner, X, Y):
        Y_hat = learner.predict(X)
        return np.array(prfs(Y, Y_hat, average='weighted')[:3])

    def improve(self, example, y):
        return self.Y[example]

    def improve_explanation(self, explainer, x_explainable, explanation):
        if explanation is None:
            return None, None, None, -1, None, None
        return explainer.explain(self, self.oracle, x_explainable)



class ReligionProblem(Problem):
    """Newsgroup problem over text documents."""
    def __init__(self, rng=None):
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        CATEGORIES = ['alt.atheism', 'soc.religion.christian']
        dataset = fetch_20newsgroups(subset='all', categories=CATEGORIES,
                                     random_state=rng)
        self.Y = dataset.target
        self.examples = list(range(len(self.Y)))

        vectorizer = CountVectorizer(lowercase=False, ngram_range=(1, 2))
        self.vectorizer = vectorizer.fit(dataset.data)
        self.X = np.array(vectorizer.transform(dataset.data).todense(),
                          dtype=np.float32)

        vectorizer = TfidfVectorizer(lowercase=False)
        self.explainable_vectorizer = vectorizer.fit(dataset.data)
        self.X_explainable = np.array(vectorizer.transform(dataset.data).todense(),
                                      dtype=np.float32)

    def evaluate(self, learner, X, Y):
        Y_hat = learner.predict(X)
        return np.array(prfs(Y, Y_hat, average='weighted')[:3])

    def improve(self, example, y, g):
        if g is not None:
            raise NotImplementedError()
        return self.Y[example], g
