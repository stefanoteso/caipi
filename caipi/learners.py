import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.gaussian_process import (GaussianProcessRegressor,
                                      GaussianProcessClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils import check_random_state

from .utils import densify, vstack, hstack


class ActiveLearner:
    def __init__(self, problem, rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

    def _check_preprocessed(self, X):
        if X.ndim != 2:
            return self.problem.preproc(X)
        return X

    def select_model(self, X, y):
        raise NotImplementedError()

    def fit(self, X, y):
        X = self._check_preprocessed(X)
        self._decision_model.fit(X, y)
        if self._decision_model is not self._prob_model:
            self._prob_model.fit(X, y)

    def get_params(self):
        try:
            return np.array(self._decision_model.coef_, copy=True)
        except AttributeError:
            return None

    def decision_function(self, X):
        X = self._check_preprocessed(X)
        return self._decision_model.decision_function(X)

    def score(self, X, y):
        X = self._check_preprocessed(X)
        return self._decision_model.score(X, y)

    def predict(self, X):
        X = self._check_preprocessed(X)
        return self._decision_model.predict(X)

    def predict_proba(self, X):
        X = self._check_preprocessed(X)
        return self._prob_model.predict_proba(X)


class LinearLearner(ActiveLearner):
    def __init__(self, *args, strategy='random', model='svm', C=None,
                 sparse=False, **kwargs):

        super().__init__(*args, **kwargs)
        self.model = model

        pm = None
        if model == 'lr':
            # logistic regression
            dm = pm = LogisticRegression(C=C or 1000,
                                         penalty='l2',
                                         multi_class='ovr',
                                         random_state=0)

        elif model == 'svm':
            # linear SVM (dense)
            dm = LinearSVC(C=C or 1000,
                           penalty='l2',
                           loss='hinge',
                           multi_class='ovr',
                           random_state=0)

        elif model == 'l1svm':
            # linear SVM (sparse)
            dm = LinearSVC(C=C or 1,
                           penalty='l1',
                           loss='squared_hinge',
                           dual=False,
                           multi_class='ovr',
                           random_state=0)

        elif model == 'elastic':
            # elastic net (kinda sparse)
            dm = SGDClassifier(penalty='elasticnet',
                               loss='hinge',
                               l1_ratio=0.15,
                               random_state=0)

        elif model == 'polysvm':
            dm = pm = SVC(C=C or 1,
                          kernel='poly',
                          probability=True,
                          decision_function_shape='ovr',
                          random_state=0)

        if pm is None:
            cv = StratifiedKFold(random_state=0)
            pm = CalibratedClassifierCV(dm, method='sigmoid', cv=cv)

        self._decision_model = dm
        self._prob_model = pm

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
        Cs = np.logspace(-3, 3, 7)

        X = self._check_preprocessed(X)
        grid = GridSearchCV(estimator=self._decision_model,
                            param_grid=dict(C=Cs),
                            scoring='f1_weighted',
                            n_jobs=-1)
        grid.fit(X, y)
        best_C = grid.best_estimator_.C
        print('SVM: setting C to', best_C)
        self._decision_model.set_params(C=best_C)
        # XXX do we have to update the prob model here?



class GPLearner(ActiveLearner):
    def __init__(self, *args, strategy='random', **kwargs):
        super().__init__(*args, **kwargs)

        self._decision_model = GaussianProcessClassifier(random_state=0)
        self._prob_model = GaussianProcessRegressor(random_state=0)

        self.select_query = {
            'random': self._select_at_random,
            'most-variance': self._select_most_variance,
        }[strategy]

    def select_model(self, X, y):
        pass

    def _select_at_random(self, problem, examples):
        return self.rng.choice(sorted(examples))

    def _select_most_variance(self, problem, examples):
        examples = sorted(examples)
        _, std = self._prob_model.predict(problem.X[examples], return_std=True)
        return examples[np.argmax(std)]
