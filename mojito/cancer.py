import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from .problems import Problem
from .utils import PipeStep, TextMod


def _poly(a, b):
    return np.array([ai*bj for ai in a for bj in b])

_POLY = PipeStep(lambda X: np.array([_poly(x, x) for x in X]))


class CancerProblem(Problem):
    """The breast cancer dataset.

    Features:
    - non explainable: 2nd degree homogeneous polynomial of the attributes
    - explainable: the attributes

    Partially ripped from https://github.com/marcotcr/lime
    """
    def __init__(self, rng=None):
        dataset = load_breast_cancer()

        self.Y = dataset.target
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(_POLY.transform(dataset.data))
        self.X_lime = scaler.fit_transform(dataset.data)
        self.examples = list(range(len(self.Y)))

        self.class_names = dataset.target_names
        self.feature_names = dataset.feature_names

    def wrap_preproc(self, model):
        return model

    def explain(self, learner, known_examples, example, y,
                num_samples=5000, num_features=10):
        explainer = LimeTabularExplainer(self.X_lime[known_examples],
                                         mode='classification',
                                         class_names=self.class_names,
                                         feature_names=self.feature_names,
                                         categorical_features=[],
                                         discretize_continuous=True,
                                         verbose=True)

        local_model = Ridge(alpha=1, fit_intercept=True)
        try:
            pipeline = make_pipeline(_POLY, learner.model_)
        except AttributeError:
            pipeline = make_pipeline(_POLY, learner)
        explanation = explainer.explain_instance(self.X_lime[example],
                                                 pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_samples=num_samples,
                                                 num_features=num_features)

        # TODO extract datapoints, coefficients, intercept, discrepancy
        explanation.discrepancy = -1
        return explanation

    def improve(self, example, y):
        return self.Y[example]

    def get_class_name(self, y):
        """Returns the colorized class name for the given label."""
        # XXX it only supports 7 distinct colors...
        CLASS_COLOR = [
            TextMod.RED,
            TextMod.GREEN,
            TextMod.BLUE,
            TextMod.PURPLE,
            TextMod.CYAN,
            TextMod.YELLOW,
            TextMod.DARKCYAN,
        ]
        return (TextMod.BOLD + CLASS_COLOR[y % len(CLASS_COLOR)] +
                self.class_names[y] + TextMod.END)

    def improve_explanation(self, example, y, explanation):

        print('The model thinks that this instance:')
        print(self.X_lime[example])
        print('is {}, because of these values:'.format(self.get_class_name(y)))
        for constraint, coeff in explanation.as_list():
            color = TextMod.RED if coeff < 0 else TextMod.GREEN
            coeff = TextMod.BOLD + color + '{:+3.1f}'.format(coeff) + TextMod.END
            print('  {:40s} : {}'.format(constraint, coeff))

        # TODO acquire improved explanation

        return explanation

    @staticmethod
    def to_range(feat):
        if ' > ' in feat:
            # 'feature > value'
            name, lb = feat.split(' > ')
            lb, ub = float(lb), np.inf
        elif ' < ' in feat:
            # 'value < feature <= value'
            name, ub = feat.split(' <= ')
            lb, name = name.split(' < ')
            lb, ub = float(lb), float(ub)
        else:
            # 'feature <= value'
            name, ub = feat.split(' <= ')
            lb, ub = -np.inf, float(ub)
        return name, (lb, ub)

    @staticmethod
    def intersect(range1, coeff1, range2, coeff2):
        lb = max(range1[0], range2[0])
        ub = min(range1[1], range2[1])
        if lb < ub and np.sign(coeff1) == np.sign(coeff2):
            return 1
        return 0

    def get_explanation_perf(self, true_explanation, pred_explanation,
                             min_coeff=1e-4):
        """Compute the explanation recall."""
        num_retrieved, num_relevant = 0, 0
        for true_feat, true_coeff in true_explanation.as_list():
            true_name, true_range = self.to_range(true_feat)
            num_relevant += int(np.abs(true_coeff) > min_coeff)
            for pred_feat, pred_coeff in pred_explanation.as_list():
                pred_name, pred_range = self.to_range(pred_feat)
                if true_name == pred_name:
                    num_retrieved += self.intersect(true_range, true_coeff,
                                                    pred_range, pred_coeff)
        return num_retrieved / num_relevant
