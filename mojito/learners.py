import numpy as np
from sklearn.utils import check_random_state


class ActiveLearner:
    def select_query(self, problem, examples):
        raise NotImplementedError('virtual method')

    def predict(self, X):
        raise NotImplementedError('virtual method')

    def fit(self, X, Y, X_lime=None, R_lime=None):
        raise NotImplementedError('virtual method')


class ActiveSVM(ActiveLearner):
    """A simple-minded implementation of active SVM.

    It simply fits a base SVM model from scratch every time a new
    example is added. This is definitely *not* efficient, but it's
    enough for our purposes.

    A calibrated classifier is used to squeeze probability estimates
    from the base SVM model using Platt scaling. These are needed
    by LIME.
    """
    def __init__(self, strategy, C=1.0, rng=None):
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        self.select_query = {
            'random': self.select_at_random_,
            'least-confident': self.select_least_confident_,
            'least-margin': self.select_least_margin_,
            'most-improvement': self.select_most_improvement_,
        }[strategy]
        self.rng = check_random_state(rng)

        self.svm_ = LinearSVC(C=C, penalty='l2', loss='hinge',
                              multi_class='ovr', random_state=rng)
        self.model_ = CalibratedClassifierCV(self.svm_, method='sigmoid')

    def fit(self, X, Y):
        # XXX unfortunately, by design, fittin the CCCV does not fit the SVM
        self.svm_.fit(X, Y)
        self.model_.fit(X, Y)

    def predict(self, X):
        return self.svm_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def select_at_random_(self, X, Y, examples):
        return self.rng.choice(list(examples))

    def select_least_confident_(self, X, Y, examples):
        """Selects the example closest to the separation hyperplane."""
        examples = list(examples)
        margins = np.abs(self.svm_.decision_function(X[examples]))
        if margins.ndim == 2:
            margins = margins.min(axis=1)
        return examples[np.argmin(margins)]

    def select_least_margin_(self, X, Y, examples):
        """Selects the example whose most likely label is closest to the second
        most-likely label."""
        examples = list(examples)
        probs = self.predict_proba(X[examples])
        prob_diffs = np.zeros(probs.shape[0])
        for i, prob in enumerate(probs):
            sorted_indices = np.argsort(prob)
            prob_diffs[i] = (prob[sorted_indices[-1]] -
                             prob[sorted_indices[-2]])
        return examples[np.argmin(prob_diffs)]

    def select_most_improvement_(self, X, Y, examples):
        raise NotImplementedError()


def ActiveGP(ActiveLearner):
    def __init__(self, strategy, kernel=None, rng=None):
        from sklearn.gaussian_process import GaussianProcessClassifier

        self.select_query = {
            'random': self.select_at_random_,
            'variance': self.select_by_variance_,
            'improvement': self.select_by_improvement_,
        }[strategy]
        self.rng = check_random_state(rng)

        self.model_ = GaussianProcessClassifier(kernel=kernel,
                                                random_state=rng)

    def fit(self, X, Y):
        self.model_.fit(X, Y)

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def select_at_random_(self, X, Y, examples):
        return self.rng.choice(list(examples))

    def select_by_variance(self, X, Y, examples):
        raise NotImplementedError()

    def select_by_improvement(self, X, Y, examples):
        raise NotImplementedError()
