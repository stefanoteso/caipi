import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import check_random_state

from .utils import densify, vstack, hstack


class SVMLearner:
    def __init__(self, problem, strategy, C=1.0, kernel='linear', sparse=False,
                 rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        cv = StratifiedKFold(random_state=0)
        if not sparse:
            self._f_model = LinearSVC(C=C,
                                      penalty='l2',
                                      loss='hinge',
                                      multi_class='ovr',
                                      random_state=0)
        else:
            self._f_model = LinearSVC(C=C,
                                      penalty='l1',
                                      loss='squared_hinge',
                                      dual=False,
                                      multi_class='ovr',
                                      random_state=0)
        self._p_model = CalibratedClassifierCV(self._f_model,
                                               method='sigmoid',
                                               cv=cv)

        self.select_query = {
            'random': self._select_at_random,
            'least-confident': self._select_least_confident,
            'least-margin': self._select_least_margin,
        }[strategy]

    def _select_at_random(self, problem, examples):
        return self.rng.choice(sorted(examples))

    def _select_least_confident(self, problem, examples):
        examples = sorted(examples)
        margins = np.abs(self.decision_function(problem.X[examples]))
        # NOTE margins has shape (n_examples,) or (n_examples, n_classes)
        if margins.ndim == 2:
            margins = margins.min(axis=1)
        return examples[np.argmin(margins)]

    def _select_least_margin(self, problem, examples):
        examples = sorted(examples)
        probs = self.predict_proba(problem.X[examples])
        # NOTE probs has shape (n_examples, n_classes)
        diffs = np.zeros(probs.shape[0])
        for i, prob in enumerate(probs):
            sorted_indices = np.argsort(prob)
            diffs[i] = prob[sorted_indices[-1]] - prob[sorted_indices[-2]]
        return examples[np.argmin(diffs)]

    def select_model(self, X, y):
        return
        if X.ndim != 2:
            X = self.problem.preproc(X)
        Cs = np.logspace(-3, 3, 7)
        grid = GridSearchCV(estimator=self._f_model,
                            param_grid=dict(C=Cs),
                            scoring='f1_weighted',
                            n_jobs=-1)
        grid.fit(X, y)
        best_C = grid.best_estimator_.C
        print('SVM: setting C to', best_C)
        self._f_model.set_params(C=best_C)

    def fit(self, X, y):
        if X.ndim != 2:
            X = self.problem.preproc(X)
        self._f_model.fit(X, y)
        self._p_model.fit(X, y)

    def get_params(self):
        return np.array(self._f_model.coef_, copy=True)

    def decision_function(self, X):
        if X.ndim != 2:
            X = self.problem.preproc(X)
        return self._f_model.decision_function(X)

    def predict(self, X):
        if X.ndim != 2:
            X = self.problem.preproc(X)
        return self._f_model.predict(X)

    def predict_proba(self, X):
        if X.ndim != 2:
            X = self.problem.preproc(X)
        return self._p_model.predict_proba(X)


class LRLearner:
    def __init__(self, problem, strategy, C=1.0, kernel='linear', rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        self._model = LogisticRegression(C=C,
                                         penalty='l2',
                                         random_state=0)

        self.select_query = {
            'random': self._select_at_random,
            'least-confident': self._select_least_confident,
        }[strategy]

    def select_model(self, X, y):

        Cs = np.logspace(-3, 3, 7)
        grid = GridSearchCV(estimator=self._model,
                            param_grid=dict(C=Cs),
                            scoring='f1_weighted',
                            n_jobs=-1)
        grid.fit(X, y)
        best_C = grid.best_estimator_.C
        print('LR: setting C to', best_C)
        self._model.set_params(C=best_C)

    def _select_at_random(self, problem, examples):
        return self.rng.choice(sorted(examples))

    def _select_least_confident(self, problem, examples):
        examples = sorted(examples)
        margins = np.abs(self.decision_function(problem.X[examples]))
        # NOTE margins has shape (n_examples,) or (n_examples, n_classes)
        if margins.ndim == 2:
            margins = margins.min(axis=1)
        return examples[np.argmin(margins)]

    def fit(self, X, y):
        self._model.fit(X, y)

    def get_params(self):
        return np.array(self._model.coef_, copy=True)

    def decision_function(self, X):
        return self._model.decision_function(X)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class GPLearner:
    def __init__(self, problem, strategy, rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        self._r_model = GaussianProcessRegressor(random_state=0)
        self._c_model = GaussianProcessClassifier(random_state=0)

        self.select_query = {
            'random': self._select_at_random,
            'most-variance': self._select_most_variance,
        }[strategy]

    def _select_at_random(self, problem, examples):
        return self.rng.choice(sorted(examples))

    def _select_most_variance(self, problem, examples):
        examples = sorted(examples)
        _, std = self._r_model.predict(problem.X[examples], return_std=True)
        return examples[np.argmax(std)]

    def fit(self, X, y):
        self._r_model.fit(X, y)
        self._c_model.fit(X, y)

    def predict(self, X):
        return self._c_model.predict(X)

    def predict_proba(self, X):
        return self._c_model.predict_proba(X)
