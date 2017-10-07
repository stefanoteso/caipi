import numpy as np
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state


class ActiveLearner:
    def select_query(self, problem, examples):
        raise NotImplementedError('virtual method')

    def fit(self, problem, examples):
        raise NotImplementedError('virtual method')

    def predict(self, X):
        raise NotImplementedError('virtual method')


class ActiveSVM(ActiveLearner):
    def __init__(self, strategy, C=1.0, rng=None):
        self.select_query = {
            'random': self.select_at_random_,
            'margin': self.select_by_margin_,
        }[strategy]
        self.rng = check_random_state(rng)

        self.model_ = LinearSVC(penalty='l2', loss='hinge', C=C,
                                random_state=rng)

    def fit(self, X, Y):
        self.model_.fit(X, Y)

    def score(self, X):
        return self.model_.decision_function(X)

    def predict(self, X):
        return self.model_.predict(X)

    def select_at_random_(self, X, Y, examples):
        return self.rng.choice(list(examples))

    def select_by_margin_(self, X, Y, examples):
        examples = list(examples)
        margin = np.abs(self.score(X[examples]))
        return examples[np.argmin(margin)]


class ActiveTandemSVM(ActiveLearner):
    # TODO: make it as robust as the sklearn implementation
    # TODO: integrate explanation feedback

    def __init__(self, strategy, C=1.0, rng=None):
        self.select_query = {
            'random': self.select_at_random_,
            'margin': self.select_by_margin_,
        }[strategy]
        self.C = C
        self.rng = check_random_state(rng)

        self.model_ = LinearSVC(penalty='l2', loss='hinge', C=C,
                                random_state=rng)

    def fit_(self, X, Y):
        import cvxpy as cvx

        num_examples, num_features = X.shape

        w = cvx.Variable(num_features)
        b = cvx.Variable()

        Y = (2 * Y - 1).astype(np.float32)

        loss = cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(Y, X*w - b)))
        objective = 1/2 * cvx.norm(w, 2) + self.C / num_examples * loss

        problem = cvx.Problem(cvx.Minimize(objective))
        # ['LS', 'ECOS_BB', 'GUROBI', 'SCS', 'ECOS']
        problem.solve(solver=cvx.ECOS, verbose=True)
        print('QP status =', problem.status)
        return np.array(w.value).ravel(), b.value

    def fit(self, X, Y):
        self.w_, self.b_ = self.fit_(X, Y)

    def score(self, X):
        return np.dot(X, self.w_.T) - self.b_

    def predict(self, X):
        return (np.sign(self.score(X)) + 1) / 2

    def select_at_random_(self, X, Y, examples):
        return self.rng.choice(list(examples))

    def select_by_margin_(self, X, Y, examples):
        examples = list(examples)
        margin = np.abs(self.score(X[examples]))
        return examples[np.argmin(margin)]
