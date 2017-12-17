import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from blessings import Terminal
from itertools import product

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
                                         discretize_continuous=False,
                                         verbose=False)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        pipeline = self.get_pipeline(learner)
        explanation = explainer.explain_instance(self.X_lime[example],
                                                 pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_samples=num_samples,
                                                 num_features=num_features)
        return explanation

    def improve(self, example, y):
        return self.y[example]

    def to_text(self, example):
        return self.X_lime[example]

    def get_class_name(self, y):
        return (_TERM.bold +
                _TERM.color(y) +
                self.class_names[y] +
                _TERM.normal)

    def improve_explanation(self, example, y, explanation):
        class_name = self.get_class_name(y)
        true_class_name = self.get_class_name(self.y[example])
        text = self.to_text(example)
        print(("The model thinks that this example is '{class_name}' "
               "(it actually is '{true_class_name}'):\n" +
               "{text}\n" +
               "because of these features:\n").format(**locals()))

        for constraint, coeff in explanation.as_list():
            color = _TERM.red if coeff < 0 else _TERM.green
            coeff = _TERM.bold + color + '{:+3.1f}'.format(coeff) + _TERM.normal
            print('  {:40s} : {}'.format(constraint, coeff))

        # TODO acquire improved explanation

        return explanation

    @staticmethod
    def to_name_range(feat):
        if ' > ' in feat:
            # 'feature > value'
            name, lb = feat.split(' > ')
            lb, ub = float(lb), np.inf
        elif ' < ' in feat:
            # 'value < feature <= value'
            name, ub = feat.split(' <= ')
            lb, name = name.split(' < ')
            lb, ub = float(lb), float(ub)
        elif ' <= ' in feat:
            # 'feature <= value'
            name, ub = feat.split(' <= ')
            lb, ub = -np.inf, float(ub)
        else:
            name = feat
            lb, ub = -np.inf, np.inf
        return name, (lb, ub)

    @staticmethod
    def intersect(range1, coeff1, range2, coeff2):
        lb = max(range1[0], range2[0])
        ub = min(range1[1], range2[1])
        if lb < ub and np.sign(coeff1) == np.sign(coeff2):
            return 1
        return 0

    def get_explanation_perf(self, true_explanation, pred_explanation):
        """Compute explanation precision, recall, and F1."""
        def to_name_range_coeff(explanation):
            return [(*self.to_name_range(feat), coeff)
                    for feat, coeff in explanation.as_list()
                    if np.abs(coeff) > self.min_coeff]

        true = to_name_range_coeff(true_explanation)
        pred = to_name_range_coeff(pred_explanation)

        num_hits = 0
        for true_name, true_range, true_coeff in true:
            for pred_name, pred_range, pred_coeff in pred:
                if (true_name == pred_name and
                    np.sign(true_coeff) == np.sign(pred_coeff)):
                    num_hits += self.intersect(true_range, true_coeff,
                                               pred_range, pred_coeff)

        pr = num_hits / len(pred)
        rc = num_hits / len(true)
        return pr, rc, 2 * pr * rc / (pr + rc + 1e-6)


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

    def get_pipeline(self, learner):
        return make_pipeline(_POLY, learner)


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

    def get_pipeline(self, learner):
        return make_pipeline(_POLY, learner)


class TicTacToeProblem(TabularProblem):
    """The tic-tac-toe dataset. Classify winning moves for x based on the
    board state.

    Uninterpretable features include all combinations of positions in the
    board.

    Interpretable features include individual board pieces.
    """
    def __init__(self, *args, **kwargs):
        from os.path import join
        from sklearn.preprocessing import MinMaxScaler

        self._boards, X_lime, y = [], [], []
        with open(join('data', 'tic-tac-toe.data'), 'rt') as fp:
            for line in map(str.strip, fp.readlines()):
                chars = line.replace('b', ' ').split(',')
                board = [[chars[3*i+j] for j in range(3)] for i in range(3)]
                self._boards.append(board)
                X_lime.append(self.to_lime_features(board))
                y.append({'positive': 1, 'negative': 0}[chars[-1]])

        X_lime = np.array(X_lime, dtype=np.float64)
        X = [self.to_features(x_lime) for x_lime in X_lime]
        y = np.array(y, dtype=np.int8)

        self.pipestep = PipeStep(lambda X_lime: np.array([
                self.to_features(x_lime) for x_lime in X_lime
            ]))

        feature_names = []
        for i, j in product(range(3), range(3)):
            for state in (' ', 'x', 'o'):
                feature_names.append('board[{i},{j}] is a "{state}"'.format(
                    **locals()))

        scaler = MinMaxScaler()
        super().__init__(*args,
                         y=y,
                         X=scaler.fit_transform(X),
                         X_lime=scaler.fit_transform(X_lime),
                         class_names=('win', 'no-win'),
                         feature_names=feature_names,
                         **kwargs)

    def to_text(self, example):
        """Turns an example into a string describing the board state."""
        return '\n'.join(map(str, self._boards[example]))

    @staticmethod
    def to_lime_features(board):
        """Turns a board into interpretable features."""
        x_lime = []
        for i, j in product(range(3), range(3)):
            for piece in (' ', 'x', 'o'):
                x_lime.append(board[i][j] == piece)
        return np.array(x_lime, dtype=np.float64)

    @staticmethod
    def to_features(x_lime):
        """Turns interpretable features into uninterpretable features.

        Note that the features do wrap around the board. It's more fair this
        way.
        """
        def is_piece_at(x_lime, i, j, piece):
            return x_lime[i*9 + j*3 + piece]

        TRIPLETS = list(product(range(3), repeat=3))

        x = []
        for i in range(3):
            x.extend([is_piece_at(x_lime, i, 0, triplet[0]) and
                      is_piece_at(x_lime, i, 1, triplet[1]) and
                      is_piece_at(x_lime, i, 2, triplet[2])
                      for triplet in TRIPLETS])
        for j in range(3):
            x.extend([is_piece_at(x_lime, 0, j, triplet[0]) and
                      is_piece_at(x_lime, 1, j, triplet[1]) and
                      is_piece_at(x_lime, 2, j, triplet[2])
                      for triplet in TRIPLETS])
        x.extend([is_piece_at(x_lime, 0, 0, triplet[0]) and
                  is_piece_at(x_lime, 1, 1, triplet[1]) and
                  is_piece_at(x_lime, 2, 2, triplet[2])
                  for triplet in TRIPLETS])
        x.extend([is_piece_at(x_lime, 0, 2, triplet[0]) and
                  is_piece_at(x_lime, 1, 1, triplet[1]) and
                  is_piece_at(x_lime, 2, 0, triplet[2])
                  for triplet in TRIPLETS])
        return x

    def get_pipeline(self, learner):
        return make_pipeline(self.pipestep, learner)
