import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from blessings import Terminal
from itertools import product

from .problems import Problem
from .utils import PipeStep


class TabularProblem(Problem):
    def __init__(self, y, X, Z, class_names, feature_names,
                 categorical_features, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.y = y
        self.X = X
        self.Z = Z
        self.class_names = class_names
        self.feature_names = feature_names
        self.categorical_features = categorical_features

        self.examples = list(range(len(self.y)))
        self.term = Terminal()

    def wrap_preproc(self, model):
        return model

    def explain(self, learner, known_examples, example, y,
                num_samples=5000, num_features=10):
        explainer = LimeTabularExplainer(self.Z[known_examples],
                                         mode='classification',
                                         class_names=self.class_names,
                                         feature_names=self.feature_names,
                                         categorical_features=self.categorical_features,
                                         discretize_continuous=False,
                                         verbose=False)

        local_model = Ridge(alpha=1000, fit_intercept=True, random_state=0)
        pipeline = self.get_pipeline(learner)
        explanation = explainer.explain_instance(self.Z[example],
                                                 pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_samples=num_samples,
                                                 num_features=num_features)
        return explanation

    def improve(self, example, y):
        return self.y[example]

    def to_text(self, example):
        return self.Z[example]

    def get_class_name(self, y):
        return (self.term.bold +
                self.term.color(y) +
                self.class_names[y] +
                self.term.normal)

    def improve_explanation(self, example, y, explanation):
        class_name = self.get_class_name(y)
        true_class_name = self.get_class_name(self.y[example])
        text = self.to_text(example)
        print(("The model thinks that this example is '{class_name}' "
               "(it actually is '{true_class_name}'):\n" +
               "{text}\n" +
               "because of these features:\n").format(**locals()))

        for constraint, coeff in explanation.as_list():
            color = self.term.red if coeff < 0 else self.term.green
            coeff = self.term.bold + color + '{:+3.1f}'.format(coeff) + self.term.normal
            print('  {:40s} : {}'.format(constraint, coeff))

        print('The explanation fidelity is:', explanation.score)

        # TODO acquire improved explanation

        return explanation

    def save_explanation(self, basename, example, y, explanation):
        pass

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
    def intersect(range1, range2):
        lb = max(range1[0], range2[0])
        ub = min(range1[1], range2[1])
        if lb < ub:
            return 1
        return 0

    def get_explanation_perf(self, true_explanation, pred_explanation):
        """Compute explanation precision, recall, and F1."""
        def to_name_range_coeff(explanation):
            return [(*self.to_name_range(feat), coeff)
                    for feat, coeff in explanation
                    if np.abs(coeff) >= self.min_coeff]

        true = to_name_range_coeff(true_explanation)
        pred = to_name_range_coeff(pred_explanation)

        num_hits = 0
        for true_name, true_range, true_coeff in true:
            for pred_name, pred_range, pred_coeff in pred:
                if (true_name == pred_name and
                    np.sign(true_coeff) == np.sign(pred_coeff)):
                    num_hits += self.intersect(true_range, pred_range)

        pr = (num_hits / len(pred)) if len(pred) else 0
        rc = (num_hits / len(true)) if len(true) else 0
        return pr, rc, 2 * pr * rc / (pr + rc + 1e-6)


class TicTacToeProblem(TabularProblem):
    """The tic-tac-toe dataset. Classify winning moves for x based on the
    board state.

    Uninterpretable features include all possible triplets of pieces.

    Interpretable features are the pieces on the board, i.e. x, o and blank.
    """
    def __init__(self, *args, **kwargs):
        from os.path import join

        self._boards, Z, y = [], [], []
        with open(join('data', 'tic-tac-toe.data'), 'rt') as fp:
            for line in map(str.strip, fp.readlines()):
                chars = line.split(',')
                board = [[chars[3*i+j] for j in range(3)] for i in range(3)]
                self._boards.append(board)
                Z.append(self.to_lime_features(board))
                y.append({'positive': 1, 'negative': 0}[chars[-1]])

        Z = np.array(Z, dtype=np.float64)
        X = np.array([self.to_features(z) for z in Z])
        y = np.array(y, dtype=np.int8)

        feature_names = []
        for i, j in product(range(3), range(3)):
            feature_names.append('board[{i},{j}]'.format(**locals()))

        all_features = list(range(Z.shape[1]))
        super().__init__(*args, y=y, X=X, Z=Z,
                         class_names=('no-win', 'win'),
                         feature_names=feature_names,
                         categorical_features=all_features,
                         **kwargs)

    def to_text(self, example):
        """Turns an example into a string describing the board state."""
        return '\n'.join(map(str, self._boards[example]))

    @staticmethod
    def to_lime_features(board):
        """Turns a board into interpretable features."""
        PIECE_TO_INT = {'x': -1, 'b': 0, 'o': 1}
        z = []
        for i, j in product(range(3), range(3)):
            z.append(PIECE_TO_INT[board[i][j]])
        return np.array(z, dtype=np.float64)

    @staticmethod
    def to_features(z):
        """Turns interpretable features into uninterpretable features."""
        def is_piece_at(z, i, j, piece):
            return 1.0 if z[i*3 + j] == piece else 0.0

        TRIPLETS = list(product([-1, 0, 1], repeat=3))

        x = []
        for i in range(3):
            x.extend([is_piece_at(z, i, 0, triplet[0]) and
                      is_piece_at(z, i, 1, triplet[1]) and
                      is_piece_at(z, i, 2, triplet[2])
                      for triplet in TRIPLETS])
        for j in range(3):
            x.extend([is_piece_at(z, 0, j, triplet[0]) and
                      is_piece_at(z, 1, j, triplet[1]) and
                      is_piece_at(z, 2, j, triplet[2])
                      for triplet in TRIPLETS])
        x.extend([is_piece_at(z, 0, 0, triplet[0]) and
                  is_piece_at(z, 1, 1, triplet[1]) and
                  is_piece_at(z, 2, 2, triplet[2])
                  for triplet in TRIPLETS])
        x.extend([is_piece_at(z, 0, 2, triplet[0]) and
                  is_piece_at(z, 1, 1, triplet[1]) and
                  is_piece_at(z, 2, 0, triplet[2])
                  for triplet in TRIPLETS])
        assert(sum(x) == (3 + 3 + 1 + 1))
        return x

    def get_pipeline(self, learner):
        pipestep = PipeStep(lambda Z: np.array([
                self.to_features(z) for z in Z
            ]))
        return make_pipeline(pipestep, learner)

    def save_explanation(self, basename, example, y, explanation):
        board = self._boards[example]

        # Convert features into a score by looking at the example
        PIECE_TO_INT = {'x': -1, 'b': 0, 'o': 1}
        score = np.zeros((3, 3))
        for feature_name, coeff in explanation.as_list():
            indices = feature_name.split('[')[-1].split(']')[0].split(',')
            value = int(feature_name.split('=')[-1])
            i, j = int(indices[0]), int(indices[1])
            if PIECE_TO_INT[board[i][j]] == value:
                score[i, j] += coeff
        print('feature scores =\n', score)

        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_subplot(111)

        # Draw the board
        for i in range(4):
            ax.plot([i, i], [0, 3], 'k')
            ax.plot([0, 3], [i, i], 'k')

        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)

        for i, j in product(range(3), range(3)):

            # Draw the piece
            if board[i][j] != 'b':
                ax.plot(j + 0.5,
                        3 - (i + 0.5),
                        board[i][j],
                        markersize=25,
                        markerfacecolor=(0, 0, 0),
                        markeredgecolor=(0, 0, 0),
                        markeredgewidth=2)

            # Draw the explanation highlight
            if np.abs(score[i][j]) >= self.min_coeff:
                color = (0, 1, 0, 0.3) if score[i,j] > 0 else (1, 0, 0, 0.3)
                ax.plot(j + 0.5,
                        3 - (i + 0.5),
                        's',
                        markersize=35,
                        markerfacecolor=color,
                        markeredgewidth=0)

        # Draw the prediction
        ax.text(0.2, 0.825,
                (self.class_names[y] + ' ' +
                 '(' + self.class_names[self.y[example]] + ')'),
                transform=ax.transAxes)

        # Save the PNG
        fig.savefig(basename + '.png', bbox_inches=0, pad_inches=0)
        plt.close(fig)

