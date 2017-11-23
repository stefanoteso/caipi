import numpy as np
from sklearn.utils import check_random_state
from .utils import densify


class ActiveLearner:
    def __init__(self, problem, strategy, rng=None):
        self.problem = problem
        self.strategy = strategy
        self.rng = check_random_state(rng)

    def select_query(self, problem, examples):
        raise NotImplementedError('virtual method')

    def predict(self, X):
        raise NotImplementedError('virtual method')

    def predict_proba(self, X):
        raise NotImplementedError('virtual method')

    def fit(self, X, Y):
        raise NotImplementedError('virtual method')


class ActiveSVM(ActiveLearner):
    """A simple-minded implementation of active SVM.

    It simply fits a base SVM model from scratch every time a new
    example is added. This is definitely *not* efficient, but it's
    enough for our purposes.

    A calibrated classifier is used to squeeze probability estimates
    out of the base SVM model using Platt scaling. These are needed
    by LIME.
    """
    def __init__(self, *args, C=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold

        # SVM learner used for classification
        self.svm_ = LinearSVC(C=C,
                              penalty='l2',
                              loss='hinge',
                              multi_class='ovr',
                              random_state=0)
        # Wrapper used for probability estimation
        kfold = StratifiedKFold(random_state=0)
        self.model_ = CalibratedClassifierCV(self.svm_,
                                             method='sigmoid',
                                             cv=kfold)

        self.select_query = {
            'random': self.select_at_random_,
            'least-confident': self.select_least_confident_,
            'least-margin': self.select_least_margin_,
        }[self.strategy]


    def fit(self, X, y):
        # XXX unfortunately fitting the calibrated classifier does not fit the
        # SVM, so we have to fit them both.
        self.svm_ = self.svm_.fit(X, y)
        self.model_ = self.model_.fit(X, y)

    def predict(self, X):
        return self.svm_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def select_at_random_(self, problem, examples):
        return self.rng.choice(list(examples))

    def select_least_confident_(self, problem, examples):
        """Selects the example closest to the separation hyperplane."""
        examples = list(examples)
        # margins is of shape (n_examples,) or (n_examples, n_classes)
        pipeline = problem.wrap_preproc(self.svm_)
        margins = np.abs(pipeline.decision_function(problem.X[examples]))
        if margins.ndim == 2:
            margins = margins.min(axis=1)
        return examples[np.argmin(margins)]

    def select_least_margin_(self, problem, examples):
        """Selects the example whose most likely label is closest to the second
        most-likely label."""
        examples = list(examples)
        pipeline = problem.wrap_preproc(self)
        probs = pipeline.predict_proba(problem.X[examples])
        prob_diffs = np.zeros(probs.shape[0])
        for i, prob in enumerate(probs):
            sorted_indices = np.argsort(prob)
            prob_diffs[i] = (prob[sorted_indices[-1]] -
                             prob[sorted_indices[-2]])
        return examples[np.argmin(prob_diffs)]


class ActiveGP(ActiveLearner):
    def __init__(self, *args, kernel=None, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn.gaussian_process import GaussianProcessClassifier
        self.model_ = GaussianProcessClassifier(kernel=kernel,
                                                n_jobs=-1,
                                                random_state=0)

        self.select_query = {
            'random': self.select_at_random_,
        }[self.strategy]

    def fit(self, X, y):
        self.model_ = self.model_.fit(X, y)

    def predict(self, X):
        return self.model_.predict(densify(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(densify(X))

    def select_at_random_(self, problem, examples):
        return self.rng.choice(list(examples))
