import numpy as np
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state


class ActiveLearner:
    def select_query(self, problem, examples):
        """Selects an (informative) example out of a collection."""
        raise NotImplementedError('virtual method')

    def fit(self, problem, examples):
        raise NotImplementedError('virtual method')

    def predict(self, X):
        raise NotImplementedError('virtual method')


class ActiveSVM(ActiveLearner):
    def __init__(self, strategy, C=0.1, rng=None):
        self.model = LinearSVC(random_state=rng)
        self.select_query = {
            'random': self._select_at_random,
            'margin': self._select_by_margin,
        }[strategy]
        self.rng = check_random_state(rng)

    def _select_at_random(self, X, Y, examples):
        return self.rng.choice(list(examples))

    def _select_by_margin(self, X, Y, examples):
        examples = list(examples)
        X_examples = X[examples]
        margin = np.abs(self.model.decision_function(X_examples))
        return examples[np.argmin(margin)]

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
