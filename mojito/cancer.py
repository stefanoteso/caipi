import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support as prfs

from .problems import Problem
from .utils import TextMod


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
        self.X = scaler.fit_transform(self.polynomial_(dataset.data))
        self.examples = list(range(len(self.Y)))

        self.class_names = dataset.target_names
        self.feature_names = dataset.feature_names

    def set_fold(self, train_examples):
        pass

    def polynomial_(self, X_explainable):
        def poly(a, b):
            return np.array([ai*bj for ai in a for bj in b])
        return np.array([poly(x, x) for x in X_explainable])

    def explain(self, learner, train_examples, example, num_samples=5000):
        # TODO pass num_samples in
        explainer = LimeTabularExplainer(self.X[examples],
                                         mode='classification',
                                         class_names=self.class_names,
                                         feature_names=self.feature_names,
                                         categorical_features=[],
                                         verbose=True)
        local_model = Ridge(alpha=1, fit_intercept=True)
        explanation = explainer.explain_instance(self.X[example],
                                                 learner.predict_proba,
                                                 model_regressor=local_model,
                                                 num_features=10)
        # TODO extract datapoints, coefficients, intercept, discrepancy
        return explanation, -1

    def improve(self, example, y):
        return self.Y[example]

    def improve_explanation(self, example, y, explanation):
        print(explanation.as_list())
        return explanation

    def evaluate(self, learner, examples):
        return prfs(self.Y[examples],
                    learner.predict(self.X[examples]),
                    average='weighted')[:3]
