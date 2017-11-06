import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from .problems import Problem
from .utils import TextMod


def _poly(a, b):
    return np.array([ai*bj for ai in a for bj in b])

def _polynomial(X):
    return np.array([_poly(x, x) for x in X])


class PolynomialTransformer:
    def fit(self, X, Y):
        pass

    def transform(self, X):
        return _polynomial(X)

    def predict_proba(self, X):
        raise NotImplementedError()


class CancerProblem(Problem):
    """The breast cancer dataset.

    Features:
    - non explainable: 2nd degree homogeneous polynomial of the attributes
    - explainable: the attributes
    """
    def __init__(self, rng=None):
        dataset = load_breast_cancer()

        scaler = MinMaxScaler()

        self.Y = dataset.target
        self.X_lime = scaler.fit_transform(dataset.data)
        self.X = scaler.fit_transform(_polynomial(dataset.data))
        self.examples = list(range(len(self.Y)))

        self.class_names = dataset.target_names
        self.feature_names = dataset.feature_names

        self.transformer = PolynomialTransformer()

    def explain(self, learner, known_examples, example, num_samples=5000):
        explainer = LimeTabularExplainer(self.X_lime[known_examples],
                                         mode='classification',
                                         class_names=self.class_names,
                                         feature_names=self.feature_names,
                                         categorical_features=[],
                                         discretize_continuous=True,
                                         verbose=True)

        local_model = Ridge(alpha=1, fit_intercept=True)
        pipeline = make_pipeline(self.transformer, learner.model_)
        explanation = explainer.explain_instance(self.X_lime[example],
                                                 pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_features=10)

        # TODO extract datapoints, coefficients, intercept, discrepancy
        explanation.discrepancy = -1
        return explanation

    def improve(self, example, y):
        return self.Y[example]

    def improve_explanation(self, example, y, explanation):
        print(explanation.as_list())
        return explanation
