import re
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from itertools import product
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_recall_fscore_support as prfs
from lime.lime_tabular import LimeTabularExplainer
from matplotlib.patches import Circle
from time import time

from . import Problem, PipeStep, densify, vstack, hstack, setprfs


_FEAT_NAME_REGEX = re.compile('[0-4],[0-4]')


class TabularProblem(Problem):
    def __init__(self, **kwargs):
        self.y = kwargs.pop('y')
        self.Z = kwargs.pop('Z')
        self.class_names = kwargs.pop('class_names')
        self.z_names = kwargs.pop('z_names')
        self.categorical_features = kwargs.pop('categorical_features',
                                               list(range(self.Z.shape[1])))
        self.discretize_features = kwargs.pop('discretize_features', False)
        self.lime_repeats = kwargs.pop('lime_repeats', 1)

        self.X = np.array([self.z_to_x(z) for z in self.Z], dtype=np.float64)
        self.explainable = set(range(len(self.y)))

        super().__init__(**kwargs)

    def z_to_x(self, z):
        """Converts an interpretable instance to an instance."""
        raise NotImplementedError()

    def z_to_y(self, z):
        """Computes the true label of an interpretable instance."""
        raise NotImplementedError()

    def z_to_expl(self, z):
        """Computes the true explanation of an interpretable instance."""
        raise NotImplementedError()

    def query_label(self, i):
        return self.y[i]

    @staticmethod
    def _to_feat_name(disc_feat):
        return _FEAT_NAME_REGEX.findall(disc_feat)[0]

    @staticmethod
    def _feat_to_bounds(feat):
        EPS = 1e-13
        if ' > ' in feat:                   # (lb, +infty)
            name, lb = feat.split(' > ')
            lb, ub = float(lb), np.inf
        elif ' < ' in feat:                 # (lb, ub]
            name, ub = feat.split(' <= ')
            lb, name = name.split(' < ')
            lb, ub = float(lb), float(ub)
        elif ' <= ' in feat:                # (-infty, ub]
            name, ub = feat.split(' <= ')
            lb, ub = -np.inf, float(ub)
        elif '=' in feat:                   # [lb, ub] ~ (lb - EPS, ub]
            name, value = feat.split('=')
            lb, ub = float(value) - EPS, float(value)
        else:                               # no discretization
            name, lb, ub = feat, -np.inf, np.inf
        return name, lb, ub

    def explain(self, learner, known_examples, i, y_pred):
        lime = LimeTabularExplainer(self.Z[known_examples],
                                    class_names=self.class_names,
                                    feature_names=self.z_names,
                                    kernel_width=self.kernel_width,
                                    categorical_features=self.categorical_features,
                                    discretize_continuous=self.discretize_features,
                                    feature_selection='forward_selection',
                                    verbose=False)

        step = PipeStep(lambda Z: np.array([self.z_to_x(z) for z in Z],
                                           dtype=np.float64))
        pipeline = make_pipeline(step, learner)


        try:
            counts = defaultdict(int)
            for r in range(self.lime_repeats):

                t = time()
                local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
                expl = lime.explain_instance(self.Z[i],
                                             pipeline.predict_proba,
                                             model_regressor=local_model,
                                             num_samples=self.n_samples,
                                             num_features=self.n_features,
                                             distance_metric=self.metric)
                print('  LIME {}/{} took {}s'.format(r + 1, self.lime_repeats,
                                                     time() - t))

                for feat, coeff in expl.as_list():
                    coeff = int(np.sign(coeff))
                    counts[(feat, coeff)] += 1

            sorted_counts = sorted(counts.items(), key=lambda _: _[-1])
            sorted_counts = list(sorted_counts)[-self.n_features:]
            return [fs for fs, _ in sorted_counts]

        except FloatingPointError:
            # XXX sometimes the calibrator classifier CV throws this
            print('Warning: LIME failed, returning no explanation')
            return None

    def _eval_expl(self, learner, known_examples, eval_examples,
                   t=None, basename=None):

        if eval_examples is None:
            return -1, -1, -1

        perfs = []
        for i in eval_examples:
            true_y = self.y[i]
            true_expl = self.z_to_expl(self.Z[i])

            pred_y = learner.predict(densify(self.X[i]))[0]
            pred_expl = self.explain(learner, known_examples, i, pred_y)
            if pred_expl is None:
                print('Warning: skipping eval example')
                return -1, -1, -1

            true_feats = {feat.split('=')[0] for feat, _ in true_expl}
            pred_feats = {self._feat_to_bounds(feat)[0] for feat, _
                          in pred_expl}
            perfs.append(setprfs(true_feats, pred_feats))

            if basename is None:
                continue

            self.save_expl(basename + '_{}_true.png'.format(i),
                           i, true_y, true_expl)
            self.save_expl(basename + '_{}_{}.png'.format(i, t),
                           i, pred_y, pred_expl)

        return np.mean(perfs, axis=0)

    def eval(self, learner, known_examples, test_examples, eval_examples,
             t=None, basename=None):
        pred_perfs = prfs(self.y[test_examples],
                          learner.predict(self.X[test_examples]),
                          average='weighted')[:3]
        expl_perfs = self._eval_expl(learner,
                                     known_examples,
                                     eval_examples,
                                     t=t, basename=basename)
        return tuple(pred_perfs) + tuple(expl_perfs)


_COORDS_FST = [[0, 0], [0, 2]]
_COORDS_LST = [[2, 0], [2, 2]]


class ToyProblem(TabularProblem):
    """A toy problem about classifying 3x3 black and white images."""

    def __init__(self, rule='fst', **kwargs):
        if not rule in ('fst', 'lst'):
            raise ValueError('invalid rule "{}"'.format(rule))

        Z = np.array(list(product([0, 1], repeat=9)))
        y = np.array(list(map(self.z_to_y, Z)))

        notxor = lambda a, b: (a and b) or (not a and not b)
        valid_examples = [i for i in range(len(y))
                          if notxor(self._rule_fst(Z[i]), self._rule_lst(Z[i]))]

        z_names = ['{},{}'.format(r, c)
                   for r, c in product(range(3), repeat=2)]

        self.rule = rule
        super().__init__(y=y[valid_examples],
                         Z=Z[valid_examples],
                         class_names=['negative', 'positive'],
                         z_names=z_names,
                         metric='hamming',
                         **kwargs)

    def z_to_x(self, z):
        return z

    @staticmethod
    def _rule_fst(z):
        return all([z[3*r+c] for r, c in _COORDS_FST])

    @staticmethod
    def _rule_lst(z):
        return all([z[3*r+c] for r, c in _COORDS_LST])

    def z_to_y(self, z):
        return 1 if self._rule_fst(z) or self._rule_lst(z) else 0

    def z_to_expl(self, z):
        z = z.reshape((3, -1))
        feat_coeff = []
        for r, c in (_COORDS_FST if self.rule == 'fst' else _COORDS_LST):
            value = z[r, c]
            feat_coeff.append(('{},{}={}'.format(r, c, value), 2*value-1))
        return feat_coeff

    def _parse_feat(self, feat):
        r = int(feat.split(',')[0])
        c = int(feat.split(',')[-1].split('=')[0])
        value = int(feat.split(',')[-1].split('=')[-1])
        return r, c, value

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl, X_test):
        if pred_expl is None or pred_y != self.y[i]:
            return X_corr, y_corr

        z = self.Z[i]
        true_feats = {feat.split('=')[0] for (feat, _) in self.z_to_expl(z)}
        pred_feats = {feat.split('=')[0] for (feat, _) in pred_expl}

        Z_new_corr = []
        for feat in pred_feats - true_feats:
            r, c, _ = self._parse_feat(feat)
            z_corr = np.array(z, copy=True)
            z_corr[3*r+c] = 1 - z_corr[3*r+c]
            if self.z_to_y(z_corr) != pred_y or tuple(z_corr) in X_test:
                continue
            Z_new_corr.append(z_corr)

        if not len(Z_new_corr):
            return X_corr, y_corr

        X_new_corr = np.array(Z_new_corr, dtype=np.float64)
        y_new_corr = np.array([pred_y for _ in Z_new_corr], dtype=np.int8)

        X_corr = vstack([X_corr, X_new_corr])
        y_corr = hstack([y_corr, y_new_corr])
        return X_corr, y_corr

    def save_expl(self, path, i, y, expl):
        z = self.Z[i].reshape((3, -1))

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')

        ax.imshow(1 - z, interpolation='nearest', cmap=plt.get_cmap('binary'))
        for feat, coeff in expl:
            r, c, value = self._parse_feat(feat)
            if z[r, c] == value:
                color = '#00FF00' if coeff > 0 else '#FF0000'
                ax.add_patch(Circle((c, r), 0.4, color=color))

        ax.text(0.5, 1.05, 'true = {} | this = {}'.format(self.y[i], y),
                horizontalalignment='center',
                transform=ax.transAxes)

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)



_COLORS = [
    (255,   0,   0), # r
    (0,   255,   0), # g
    (0,   128, 255), # b
    (128,   0, 255), # v
]

class ColorsProblem(TabularProblem):
    """Colors problem from the "Right for the Right Reasons" paper."""

    def __init__(self, rule=0, n_examples=1000, **kwargs):
        if not rule in (0, 1):
            raise ValueError('invalid rule "{}"'.format(rule))
        self.rule = rule

        data = np.load(join('data', 'toy_colors.npz'))
        images = np.vstack([data['arr_0'], data['arr_1']])
        images = np.array([image.reshape((5, 5, 3)) for image in images])
        labels = 1 - np.hstack([data['arr_2'], data['arr_3']])

        if n_examples:
            rng = np.random.RandomState(0)
            pi = rng.permutation(len(images))[:n_examples]
        else:
            pi = list(range(len(images)))

        z_names = ['{},{}'.format(r, c)
                   for r, c in product(range(5), repeat=2)]

        super().__init__(y=np.array([labels[i] for i in pi]),
                         Z=np.array([self._image_to_z(images[i]) for i in pi]),
                         class_names=['negative', 'positive'],
                         z_names=z_names,
                         metric='hamming',
                         discretize_features=True,
                         categorical_features=[],
                         **kwargs)

    @staticmethod
    def _image_to_z(image):
        return np.array([_COLORS.index(tuple(image[r, c]))
                         for r, c in product(range(5), repeat=2)],
                        dtype=np.float64)

    def z_to_x(self, z):
        x = [1 if z[i] == z[j] else 0
             for i in range(5*5)
             for j in range(i+1, 5*5)]
        return np.array(x, dtype=np.float64)

    @staticmethod
    def _rule0(z):
        return z[0,0] == z[0,4] and z[0,0] == z[4,0] and z[0,0] == z[4,4]

    @staticmethod
    def _rule1(z):
        return z[0,1] != z[0,2] and z[0,1] != z[0,3] and z[0,2] != z[0,3]

    def z_to_y(self, z):
        z = z.reshape((5, 5))
        return self._rule0(z) if self.rule == 0 else self._rule1(z)

    def z_to_expl(self, z):
        z = z.reshape((5, 5))

        if self.rule == 0:
            COORDS = [[0, 0], [0, 4], [4, 0], [4, 4]]
        else:
            COORDS = [[0, 1], [0, 2], [0, 3]]

        counts = np.bincount([z[r,c] for r, c in COORDS])
        max_count, max_value = np.max(counts), np.argmax(counts)

        feat_to_coeff = defaultdict(int)
        if self.rule == 0:
            for r, c in COORDS:
                weight = 1 if max_count != 1 and z[r, c] == max_value else -1
                feat_to_coeff['{},{}={}'.format(r, c, int(z[r, c]))] += weight
        else:
            for r, c in COORDS:
                weight = 1 if max_count == 1 or z[r, c] != max_value else -1
                feat_to_coeff['{},{}={}'.format(r, c, int(z[r, c]))] += weight

        return list(feat_to_coeff.items())

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl, X_test):
        if pred_expl is None or pred_y != self.y[i]:
            return X_corr, y_corr

        z = self.Z[i]
        true_feats = {feat for (feat, _) in self.z_to_expl(z)}
        pred_feats = {feat for (feat, _) in pred_expl}

        ALL_VALUES = set(range(4))

        print(z.reshape((5, 5)))
        Z_new_corr = []
        for feat in pred_feats - true_feats:
            feat, lb, ub = self._feat_to_bounds(feat)
            r, c = feat.split(',')
            r, c = int(r), int(c)
            other_values = {value for value in ALL_VALUES if not (lb < value <= ub)}
            print(other_values)
            for value in other_values:
                z_corr = np.array(z, copy=True)
                z_corr[5*r+c] = value
                print(z_corr.reshape((5, 5)))
                if self.z_to_y(z_corr) != pred_y:
                    continue
                if tuple(z_corr) in X_test:
                    continue
                Z_new_corr.append(z_corr)

        if not len(Z_new_corr):
            return X_corr, y_corr

        X_new_corr = np.array([self.z_to_x(z_corr) for z_corr in Z_new_corr],
                              dtype=np.float64)
        y_new_corr = np.array([pred_y for _ in Z_new_corr], dtype=np.int8)

        X_corr = vstack([X_corr, X_new_corr])
        y_corr = hstack([y_corr, y_new_corr])
        return X_corr, y_corr

    def save_expl(self, path, i, pred_y, expl):
        z = self.Z[i]
        image = np.array([_COLORS[int(value)] for value in z]).reshape((5, 5, 3))

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')

        z = z.reshape((5, 5))
        ax.imshow(image, interpolation='nearest')
        for feat, coeff in expl:
            feat, lb, ub = self._feat_to_bounds(feat)
            r, c = feat.split(',')
            r, c = int(r), int(c)
            if lb < z[r, c] <= ub:
                color = '#FFFFFF' if coeff > 0 else '#000000'
                ax.add_patch(Circle((c, r), 0.35, color=color))

        ax.text(0.5, 1.05, 'true = {} | pred = {}'.format(self.y[i], pred_y),
                horizontalalignment='center',
                transform=ax.transAxes)

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)



_TRIPLETS = [
   [[0, 0], [0, 1], [0, 2]],
   [[1, 0], [1, 1], [1, 2]],
   [[2, 0], [2, 1], [2, 2]],
   [[0, 0], [1, 0], [2, 0]],
   [[0, 1], [1, 1], [2, 1]],
   [[0, 2], [1, 2], [2, 2]],
   [[0, 0], [1, 1], [2, 2]],
   [[0, 2], [1, 1], [2, 0]],
]

_SALIENT_CONFIGS = [
    # Win configs
    (( 1,  1,  1),  1),
    # Almost-win configs
    (( 1,  1,  0), -1),
    (( 1,  0,  1), -1),
    (( 0,  1,  1), -1),
    (( 1,  1, -1), -1),
    (( 1, -1,  1), -1),
    ((-1,  1,  1), -1),
]

class TTTProblem(TabularProblem):
    """Tic-tac-toe endgames."""

    def __init__(self, **kwargs):
        Z, y = [], []
        with open(join('data', 'tic-tac-toe.data'), 'rt') as fp:
            for line in map(str.strip, fp):
                chars = line.split(',')
                Z.append([{'x': 1, 'b': 0, 'o': -1}[c] for c in chars[:-1]])
                y.append({'positive': 1, 'negative': 0}[chars[-1]])
        Z = np.array(Z, dtype=np.float64)
        y = np.array(y, dtype=np.int8)

        class_names = ['no-win', 'win']
        z_names = []
        for r, c in product(range(3), repeat=2):
            z_names.append('{r},{c}'.format(**locals()))

        super().__init__(Z, y, class_names, z_names, **kwargs)

    @staticmethod
    def get_config(z, triplet):
        return tuple(int(z[3*r+c]) for r, c in triplet)

    def z_to_y(self, z):
        for triplet in _TRIPLETS:
            if self.get_config(z, triplet) == (1, 1, 1):
                return True
        return False

    def z_to_expl(self, z):
        feat_coeff = set()
        for triplet in _TRIPLETS:
            config = self.get_config(z, triplet)
            for salient_config, coeff in _SALIENT_CONFIGS:
                if config == tuple(salient_config):
                    for r, c in triplet:
                        value = int(z[3*r+c])
                        if (value == 1 if coeff else value != 1):
                            feat = '{r},{c}={value}'.format(**locals())
                            feat_coeff.add((feat, coeff))
        print(self.z_to_y(z))
        print(z.reshape((3, 3)))
        print(feat_coeff)
        quit()
        return feat_coeff

    @staticmethod
    def z_to_x(z):
        CONFIGS = list(product([-1, 0, 1], repeat=3))

        def is_piece_at(z, i, j, piece):
            return 1 if z[i*3 + j] == piece else 0

        x = []
        for i in range(3):
            x.extend([is_piece_at(z, i, 0, config[0]) and
                      is_piece_at(z, i, 1, config[1]) and
                      is_piece_at(z, i, 2, config[2])
                      for config in CONFIGS])
        for j in range(3):
            x.extend([is_piece_at(z, 0, j, config[0]) and
                      is_piece_at(z, 1, j, config[1]) and
                      is_piece_at(z, 2, j, config[2])
                      for config in CONFIGS])
        x.extend([is_piece_at(z, 0, 0, config[0]) and
                  is_piece_at(z, 1, 1, config[1]) and
                  is_piece_at(z, 2, 2, config[2])
                  for config in CONFIGS])
        x.extend([is_piece_at(z, 0, 2, config[0]) and
                  is_piece_at(z, 1, 1, config[1]) and
                  is_piece_at(z, 2, 0, config[2])
                  for config in CONFIGS])

        assert(sum(x) == (3 + 3 + 1 + 1))
        return np.array(x, dtype=np.float64)

    def query_label(self, i):
        return self.y[i]

    def query_improved_expl(self, i, pred_y, pred_z):
        true_y = self.y[i]
        if pred_y != true_y:
            return None, None

        raise NotImplementedError()

        board = self.boards[i]
        true_feats = [feat for (feat, coeff) in
                      self._board_to_expl(self.boards[i])]
        pred_feats = [feat for (feat, coeff) in pred_z]

        alt_boards = []
        for feat in set(pred_feats) - set(true_feats):
            indices = feat.split('[')[-1].split(']')[0].split(',')
            i, j = int(indices[0]), int(indices[1])
            for alt_piece in set(['o', 'b', 'x']) - set([str(board[i, j])]):
                alt_board = np.array(board)
                alt_board[i,j] = alt_piece
                # Do not add board with a wrong label
                if true_y == self._board_to_y(alt_board):
                    alt_boards.append(alt_board)
        if not len(alt_boards):
            return None, None

        X_extra = [self._z_to_x(self._board_to_z(alt_board))
                   for alt_board in alt_boards]
        y_extra = [pred_y for alt_board in alt_boards]

        return (np.array(X_extra, dtype=np.float64),
                np.array(y_extra, dtype=np.int8))

    def _score_features(self, board, expl):
        scores = np.zeros((3, 3))
        for feat, coeff in expl:
            indices = feat.split('[')[-1].split(']')[0].split(',')
            value = int(feat.split('=')[-1])
            i, j = int(indices[0]), int(indices[1])
            if self._PIECE_TO_INT[board[i,j]] == value:
                scores[i, j] += coeff
        return scores

    def save_expl(self, path, i, y, z):
        board = self.boards[i]
        scores = self._score_features(board, z)

        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_subplot(111)

        for i in range(4):
            ax.plot([i, i], [0, 3], 'k')
            ax.plot([0, 3], [i, i], 'k')

        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)

        for i, j in product(range(3), range(3)):
            if board[i, j] != 'b':
                ax.plot(j + 0.5,
                        3 - (i + 0.5),
                        board[i, j],
                        markersize=25,
                        markerfacecolor=(0, 0, 0),
                        markeredgecolor=(0, 0, 0),
                        markeredgewidth=2)
            if np.abs(scores[i][j]) > 0:
                color = (0, 1, 0, 0.3) if scores[i,j] > 0 else (1, 0, 0, 0.3)
                ax.plot(j + 0.5,
                        3 - (i + 0.5),
                        's',
                        markersize=35,
                        markerfacecolor=color,
                        markeredgewidth=0)

        ax.text(0.5, 0.825, 'y = {}'.format(y),
                horizontalalignment='center',
                transform=ax.transAxes)

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)
