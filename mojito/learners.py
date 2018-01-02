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

    def fit(self, X, y):
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
    def __init__(self, *args, C=1.0, kernel='linear',  **kwargs):
        super().__init__(*args, **kwargs)

        if kernel == 'linear':
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import StratifiedKFold

            self.scoring_model_ = LinearSVC(C=C,
                                            penalty='l2',
                                            loss='hinge',
                                            multi_class='ovr',
                                            random_state=0)

            cv = StratifiedKFold(random_state=0)
            self.probabilistic_model_ = \
                CalibratedClassifierCV(self.scoring_model_,
                                       method='sigmoid',
                                       cv=cv)
        else:
            from sklearn.svm import SVC

            self.scoring_model_ = \
            self.probabilistic_model_ = SVC(C=C,
                                            kernel=kernel,
                                            probability=True,
                                            decision_function_shape='ovr',
                                            random_state=0)

        self.select_query = {
            'random': self.select_at_random_,
            'least-confident': self.select_least_confident_,
            'least-margin': self.select_least_margin_,
        }[self.strategy]

    def decision_function(self, X):
        return self.scoring_model_.decision_function(X)

    def fit(self, X, y):
        self.scoring_model_.fit(X, y)
        if self.scoring_model_ is not self.probabilistic_model_:
            self.probabilistic_model_.fit(X, y)

    def predict(self, X):
        return self.scoring_model_.predict(X)

    def predict_proba(self, X):
        return self.probabilistic_model_.predict_proba(X)

    def select_at_random_(self, problem, examples):
        return self.rng.choice(list(examples))

    def select_least_confident_(self, problem, examples):
        """Selects the example closest to the separation hyperplane."""
        examples = list(examples)
        # margins is of shape (n_examples,) or (n_examples, n_classes)
        pipeline = problem.wrap_preproc(self)
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


class TicTacToeLearner(ActiveLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        WEIGHTS_FOR_TRIPLET = [
                1, 0, 0, # x x *
                0, 0, 0, # x b *
                0, 0, 0, # x o *
                0, 0, 0, # b x *
                0, 0, 0, # b b *
                0, 0, 0, # b o *
                0, 0, 0, # o x *
                0, 0, 0, # o b *
                0, 0, 0, # o o *
            ]

        w = []
        for i in range(3):
            w.extend(WEIGHTS_FOR_TRIPLET)
        for j in range(3):
            w.extend(WEIGHTS_FOR_TRIPLET)
        w.extend(WEIGHTS_FOR_TRIPLET)
        w.extend(WEIGHTS_FOR_TRIPLET)
        self.w = np.array(w)

        self.select_query = {
            'random': self.select_at_random_,
        }[self.strategy]

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.sign(np.dot(X, self.w)).astype(int)

    def predict_proba(self, X):
        y_hat = np.sign(np.dot(X, self.w))
        p = []
        for sign in y_hat:
            p.append([1, 0] if sign < 0.5 else [0, 1])
        p = np.array(p)
        print('LIME sample balance =', y_hat.mean())
        return p

    def select_at_random_(self, problem, examples):
        return self.rng.choice(list(examples))


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
