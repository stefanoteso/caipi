import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from blessings import Terminal

from .problems import Problem
from .utils import PipeStep


_TERM = Terminal()

def _poly(a, b):
    return np.array([ai*bj for ai in a for bj in b])

_POLY = PipeStep(lambda X: np.array([_poly(x, x) for x in X]))


class TabularProblem(Problem):
    def __init__(self, y, X, X_lime, class_names, feature_names, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.y, self.X, self.X_lime = y, X, X_lime
        self.examples = list(range(len(self.y)))
        self.class_names = class_names
        self.feature_names = feature_names

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
                                         verbose=False)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        pipeline = make_pipeline(_POLY, learner)
        explanation = explainer.explain_instance(self.X_lime[example],
                                                 pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_samples=num_samples,
                                                 num_features=num_features)
        return explanation

    def improve(self, example, y):
        return self.y[example]

    def improve_explanation(self, example, y, explanation):
        class_name = (_TERM.bold +
                      _TERM.color(y) +
                      self.class_names[y] +
                      _TERM.normal)

        x = self.X_lime[example]
        print(("The model thinks that this example is '{class_name}':\n" +
               "{x}\n" +
               "because of these features:\n").format(**locals()))

        for constraint, coeff in explanation.as_list():
            color = _TERM.red if coeff < 0 else _TERM.green
            coeff = _TERM.bold + color + '{:+3.1f}'.format(coeff) + _TERM.normal
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

    def get_explanation_perf(self, true_explanation, pred_explanation):
        """Compute the explanation recall."""
        num_retrieved, num_relevant = 0, 0
        for true_feat, true_coeff in true_explanation.as_list():
            true_name, true_range = self.to_range(true_feat)
            num_relevant += int(np.abs(true_coeff) > self.min_coeff)
            for pred_feat, pred_coeff in pred_explanation.as_list():
                pred_name, pred_range = self.to_range(pred_feat)
                if true_name == pred_name:
                    num_retrieved += self.intersect(true_range, true_coeff,
                                                    pred_range, pred_coeff)
        return num_retrieved / num_relevant


class IrisProblem(TabularProblem):
    """The iris dataset.

    Features:
    - non explainable: 2nd degree homogeneous polynomial of the attributes
    - explainable: axis-aligned constraints on the attributes
    """
    def __init__(self, *args, **kwargs):
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import MinMaxScaler

        dataset = load_iris()
        scaler = MinMaxScaler()
        super().__init__(*args,
                         y=dataset.target,
                         X=scaler.fit_transform(_POLY.transform(dataset.data)),
                         X_lime=scaler.fit_transform(dataset.data),
                         class_names=dataset.target_names,
                         feature_names=dataset.feature_names,
                         **kwargs)


class CancerProblem(TabularProblem):
    """The breast cancer dataset.

    Features:
    - non explainable: 2nd degree homogeneous polynomial of the attributes
    - explainable: axis-aligned constraints on the attributes

    Partially ripped from https://github.com/marcotcr/lime
    """
    def __init__(self, *args, **kwargs):
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import MinMaxScaler

        dataset = load_breast_cancer()
        scaler = MinMaxScaler()
        super().__init__(*args,
                         y=dataset.target,
                         X=scaler.fit_transform(_POLY.transform(dataset.data)),
                         X_lime=scaler.fit_transform(dataset.data),
                         class_names=dataset.target_names,
                         feature_names=dataset.feature_names,
                         **kwargs)

class TicTacToeProblem(TabularProblem):
    """The tic-tac-toe dataset. Classify winning moves for x based on the
    board state.
    """
    def __init__(self, *args, **kwargs):
        from os.path import join

        TO_X = {'b': 0, 'x': 1, 'o': -1}
        TO_Y = {'positive': 1, 'negative': 0}

        X, y = [], []
        with open(join('data', 'tic-tac-toe.data'), 'rt') as fp:
            for line in map(str.strip, fp.readlines()):
                chars = line.split(',')
                X.append([TO_X[char] for char in chars[:-1]])
                y.append(TO_Y[chars[-1]])

        X, y = np.array(X), np.array(y)

        super().__init__(*args,
                         y=y,
                         X=_POLY.transform(X),
                         X_lime=X,
                         class_names=('win', 'no-win'),
                         feature_names=('tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br'),
                         **kwargs)
