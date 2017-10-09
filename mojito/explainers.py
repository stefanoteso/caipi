import numpy as np
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state


class Explainer:
    """An explainer.

    Parameters
    ----------
    problem : mojito.Problem
        The problem.
    rng : numpy.RandomState, defaults to None
        The RNG.
    """
    def __init__(self, problem, rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

    def explain(self, learner, x_explainable):
        raise NotImplementedError('virtual method')


class LimeExplainer(Explainer):
    """The LIME explainer.

    Parameters
    ----------

    num_samples : int, defaults to 100
        Number of examples to sample around the prediction
    num_features : int, defaults to 10
        Number of non-zero features in the explanation.
    invsigma : float, defaults to 1
        Spread of the Gaussian example weights

    All other parameters are passed to mojito.Explainer.
    """
    def __init__(self, *args, **kwargs):
        self.num_samples = kwargs.pop('num_samples', 100)
        self.num_features = kwargs.pop('num_features', 10)
        self.invsigma = kwargs.pop('invsigma', 10000.0)
        super().__init__(*args, **kwargs)

    def _sample_dataset(self, x_explainable):
        """Samples examples around x."""
        nonzero_entries = np.flatnonzero(x_explainable)

        Z_explainable = []
        for i in range(self.num_samples):
            self.rng.shuffle(nonzero_entries)
            num_nonzeros_in_z = self.rng.randint(0, len(nonzero_entries))
            chosen = nonzero_entries[:max(1, num_nonzeros_in_z)]
            assert np.abs(x_explainable[chosen]).sum() > 0

            z_explainable = np.zeros_like(x_explainable)
            z_explainable[chosen] = x_explainable[chosen]
            Z_explainable.append(z_explainable)
        Z_explainable = np.array(Z_explainable)

        diff = x_explainable - Z_explainable
        diff = diff / np.sum(diff, axis=0)
        dist = np.diag(np.dot(diff, diff.T))
        weights = np.exp(-self.invsigma * dist**2)

        return Z_explainable, weights

    def explain(self, problem, learner, x_explainable):
        """Explains a prediction using LIME."""
        Z_explainable, w_sample = self._sample_dataset(x_explainable)
        X_explainable = problem.e2u(Z_explainable)
        Y_hat = learner.predict(X_explainable)

        print('Y_lime balance =', Y_hat.sum() / len(Y_hat))

        # TODO
        # - select first K weights using LASSO
        # - learn actual weights using least squares
        model = SVC(kernel='linear', random_state=self.rng) \
                    .fit(Z_explainable, Y_hat, sample_weight=w_sample)
        Y_explainable = model.predict(Z_explainable)
        discrepancy = np.dot(w_sample, (Y_hat - Y_explainable)**2)

        v, c = model.coef_.ravel(), model.intercept_
        indices = np.argsort(v)[-self.num_features:]
        explanation = np.zeros_like(v)
        explanation[indices] = v[indices]

        return explanation, v, c, discrepancy, Z_explainable
