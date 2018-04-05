#!/usr/bin/env python3

import numpy as np
import scipy as sp
import re
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support as prfs
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
from itertools import product
from os.path import join
from blessings import Terminal
from time import time

# TODO:
#
# - toy
#  - DONE
#
# - colors
#  - check if LIME+DT works
#  - use anchors
#
# - ttt
#  - check if LIME+DT works
#  - use anchors
#
# - newsgroups
#  - implement RBF SVM
#  - restrict queries and eval to examples with explanations
#  - move preparation script
#
# - reviews
#  - implement RBF SVM
#  - fix explanation feedback
#  - implement pos/neg feedback
#
# - images
#  - find a decent model and dataset


_TERM = Terminal()


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)


def _densify(x):
    try:
        x = x.toarray()
    except AttributeError:
        pass
    if x.shape[0] != 1:
        # if X[i] is already dense, densify(X[i]) is a no-op, so we get an x
        # of shape (n_features,) and we turn it into (1, n_features);
        # if X[i] is sparse, densify(X[i]) gives an x of shape (1, n_features).
        x = x[np.newaxis, ...]
    return x


def _stack(arrays, d_stack, s_stack):
    arrays = [a for a in arrays if a is not None]
    if len(arrays) == 0:
        return None
    if len(arrays) == 1:
        return arrays[0]
    if isinstance(arrays[0], sp.sparse.csr_matrix):
        return s_stack(arrays)
    else:
        return d_stack(arrays)

vstack = lambda arrays: _stack(arrays, np.vstack, sp.sparse.vstack)
hstack = lambda arrays: _stack(arrays, np.hstack, sp.sparse.hstack)


def _enum_nontest(problem, test_examples, X):
    test_instances = {tuple(x) for x in problem.X[test_examples]}
    return [i for i in range(len(X)) if tuple(X[i]) not in test_instances]


class PipeStep:
    def __init__(self, func):
        self.func = func

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return self.func(X)


class Problem:
    def __init__(self, n_samples, n_features, kernel_width, metric='euclidean',
                 rng=None):
        self.rng = check_random_state(rng)

        self.n_samples = n_samples
        self.n_features = n_features
        self.kernel_width = kernel_width
        self.metric = metric

    def explain(self, learner, known_examples, i, y_pred):
        """Computes the learner's explanation of a prediction."""
        raise NotImplementedError()

    def query_label(self, i):
        """Queries the oracle for a label."""
        raise NotImplementedError()

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl):
        """Queries the oracle for an improved explanation."""
        raise NotImplementedError()

    def save_expl(self, path, i, pred_y, expl):
        """Saves an explanation to file."""
        raise NotImplementedError()

    def eval(self, learner, known_examples, test_examples, eval_examples,
             t=None, basename=None):
        """Evaluates the learner."""
        raise NotImplementedError()


class TabularProblem(Problem):
    def __init__(self, *args, Z, y, class_names, z_names, **kwargs):
        self.Z, self.y = Z, y
        self.X = np.array([self.z_to_x(z) for z in Z], dtype=np.float64)
        self.class_names = class_names
        self.z_names = z_names

        super().__init__(*args, **kwargs)

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

    def explain(self, learner, known_examples, i, y_pred):
        all_features = list(range(self.Z.shape[1]))
        lime = LimeTabularExplainer(self.Z[known_examples],
                                    class_names=self.class_names,
                                    feature_names=self.z_names,
                                    kernel_width=self.kernel_width,
                                    categorical_features=all_features,
                                    discretize_continuous=False,
                                    feature_selection='forward_selection',
                                    verbose=False)

        step = PipeStep(lambda Z: np.array([self.z_to_x(z) for z in Z],
                                           dtype=np.float64))
        pipeline = make_pipeline(step, learner)

        t = time()
        # XXX set alpha to 1000 for TTT
        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        try:
            explanation = lime.explain_instance(self.Z[i],
                                                pipeline.predict_proba,
                                                model_regressor=local_model,
                                                num_samples=self.n_samples,
                                                num_features=self.n_features,
                                                distance_metric=self.metric)
            print('LIME took', time() - t, 's, score =', explanation.score)
        except FloatingPointError:
            # XXX sometimes the calibrator classifier CV throws this
            explanation = None
        return explanation

    def _eval_expl(self, learner, known_examples, eval_examples,
                   t=None, basename=None):

        if eval_examples is None:
            return -1, -1, -1

        perfs = []
        for i in eval_examples:
            true_y = self.y[i]
            pred_y = learner.predict(_densify(self.X[i]))[0]

            true_expl = self.z_to_expl(self.Z[i])
            pred_expl = self.explain(learner, known_examples, i, pred_y)
            if pred_expl is None:
                print('Warning: skipping eval example')
                return -1, -1, -1
            pred_expl = [(feat, int(np.sign(coeff)))
                         for feat, coeff in pred_expl.as_list()]

            matches = set(true_expl).intersection(set(pred_expl))
            pr = len(matches) / len(pred_expl) if len(pred_expl) else 0.0
            rc = len(matches) / len(true_expl) if len(true_expl) else 0.0
            f1 = 0.0 if pr + rc <= 0 else 2 * pr * rc / (pr + rc)
            perfs.append((pr, rc, f1))

            if basename is None:
                continue

            self.save_expl(basename + '_{}_{}.png'.format(i, t),
                           i, pred_y, pred_expl)
            self.save_expl(basename + '_{}_true.png'.format(i),
                           i, true_y, true_expl)

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

    def __init__(self, *args, rule='fst', **kwargs):
        if not rule in ('fst', 'lst'):
            raise ValueError('invalid rule "{}"'.format(rule))
        self.rule = rule

        Z = np.array(list(product([0, 1], repeat=9)))
        y = np.array(list(map(self.z_to_y, Z)))

        notxor = lambda a, b: (a and b) or (not a and not b)

        valid_examples = [i for i in range(len(y))
                          if notxor(self._rule_fst(Z[i]), self._rule_lst(Z[i]))]
        Z = Z[valid_examples]
        y = y[valid_examples]

        class_names = ['negative', 'positive']
        z_names = ['{},{}'.format(r, c)
                   for r, c in product(range(3), repeat=2)]

        super().__init__(*args, Z, y, class_names, z_names, *args,
                         metric='hamming', **kwargs)

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
        if pred_expl is None:
            return X_corr, y_corr

        z = self.Z[i]
        true_feats = [feat.split('=')[0] for (feat, _) in self.z_to_expl(z)]
        pred_feats = [feat.split('=')[0] for (feat, _) in pred_expl.as_list()]

        Z_new_corr = []
        for feat in set(pred_feats) - set(true_feats):
            r, c, _ = self._parse_feat(feat)
            z_corr = np.array(z, copy=True)
            z_corr[3*r+c] = 1 - z_corr[3*r+c]
            if self.z_to_y(z_corr) == pred_y:
                Z_new_corr.append(z_corr)

        X_new_corr = np.array([z_corr for z_corr in Z_new_corr
                               if not tuple(z_corr) in X_test],
                              dtype=np.float64)
        y_new_corr = np.array([pred_y for _ in Z_corr], dtype=np.int8)

        if not len(X_new_corr):
            return X_corr, y_corr

        X_corr = vstack([X_corr, X_new_corr])
        y_corr = hstack([y_corr, y_new_corr])
        raise NotImplementedError()
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

    def __init__(self, *args, rule=0, n_examples=1000, **kwargs):
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

        def image_to_z(image):
            return np.array([_COLORS.index(tuple(image[r, c]))
                             for r, c in product(range(5), repeat=2)],
                            dtype=np.float64)

        Z = np.array([image_to_z(images[i]) for i in pi])
        y = np.array([labels[i] for i in pi])

        class_names = ['negative', 'positive']
        z_names = ['{},{}'.format(r, c)
                   for r, c in product(range(5), repeat=2)]

        super().__init__(Z, y, class_names, z_names, *args,
                         metric='hamming', **kwargs)

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

    def _parse_feat(self, feat):
        r = int(feat.split(',')[0])
        c = int(feat.split(',')[-1].split('=')[0])
        value = int(feat.split(',')[-1].split('=')[-1])
        return r, c, value

    def query_improved_expl(self, i, pred_y, pred_expl):
        true_y = self.y[i]
        if pred_y != true_y:
            return None, None

        z = self.Z[i]
        true_feats = [feat.split('=')[0] for (feat, _) in self.z_to_expl(z)]
        pred_feats = [feat.split('=')[0] for (feat, _) in pred_expl.as_list()]

        ALL_VALUES = set(range(4))

        Z_corr = []
        for feat in set(pred_feats) - set(true_feats):
            r, c, _ = self._parse_feat(feat)
            for other_value in ALL_VALUES - set([z[5*r+c]]):
                z_corr = np.array(z, copy=True)
                z_corr[5*r+c] = other_value
                if self.z_to_y(z_corr) == pred_y:
                    Z_corr.append(z_corr)

        if not len(Z_corr):
            return None, None

        X_corr = np.array([self.z_to_x(z_corr) for z_corr in Z_corr],
                          dtype=np.float64)
        y_corr = np.array([pred_y for _ in Z_corr], dtype=np.int8)
        return X_corr, y_corr

    def save_expl(self, path, i, pred_y, expl):
        z = self.Z[i]
        image = np.array([_COLORS[int(value)] for value in z]).reshape((5, 5, 3))

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')

        z = z.reshape((5, 5))
        ax.imshow(image, interpolation='nearest')
        for feat, coeff in expl:
            r, c, value = self._parse_feat(feat)
            if z[r, c] == value:
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

    def __init__(self, *args, **kwargs):
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

        super().__init__(Z, y, class_names, z_names, *args, **kwargs)

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
        pred_feats = [feat for (feat, coeff) in pred_z.as_list()]

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



class TextProblem(Problem):
    def explain(self, learner, known_examples, i, y_pred):
        explainer = LimeTextExplainer(class_names=self.class_names)

        pipeline = make_pipeline(self.vectorizer, learner)
        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        expl = explainer.explain_instance(self.processed_docs[i],
                                          pipeline.predict_proba,
                                          model_regressor=local_model,
                                          num_features=self.n_features,
                                          num_samples=self.n_samples)
        return expl

    def query_label(self, i):
        return self.y[i]

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl, X_test):
        if pred_expl is None:
            return X_corr, y_corr

        true_words = {word for word, _ in self.explanations[i]}
        if not len(true_words):
            # No explanation known for this example
            return X_corr, y_corr
        pred_words = {word for word, _ in pred_expl.as_list()}

        doc_words = set(self.processed_docs[i].split())

        corrected_docs = []
        for word in pred_words - true_words:
            corrected_docs.append(' '.join(doc_words - set([word])))

        if not len(corrected_docs):
            return X_corr, y_corr

        X_new_corr = self.vectorizer.transform(corrected_docs)
        y_new_corr = np.array([pred_y for _ in corrected_docs], dtype=np.int8)

        X_corr = vstack([X_corr, X_new_corr])
        y_corr = hstack([y_corr, y_new_corr])
        return X_corr, y_corr

    @staticmethod
    def _highlight_words(text, expl):
        for word, coeff in expl:
            color = _TERM.green if coeff >= 0 else _TERM.red
            colored_word = color + word + _TERM.normal
            matches = list(re.compile(word).finditer(text))
            matches.reverse()
            for match in matches:
                start = match.start()
                text = text[:start] + colored_word + text[start+len(word):]
        return text

    def save_expl(self, path, i, pred_y, expl):
        with open(path, 'wt') as fp:
            fp.write('true y: ' + self.class_names[self.y[i]] + '\n')
            fp.write('pred y: ' + self.class_names[pred_y] + '\n')
            fp.write(80 * '-' + '\n')
            fp.write(self._highlight_words(self.docs[i], expl))
            fp.write('\n' + 80 * '-' + '\n')
            fp.write('explanation:\n')
            for word, coeff in expl:
                fp.write('{:32s} : {:3.1f}\n'.format(word, coeff))

    def _eval_expl(self, learner, known_examples, eval_examples,
                   t=None, basename=None):
        if eval_examples is None:
            return -1, -1, -1

        perfs = []
        for i in eval_examples:
            true_y = self.y[i]
            true_expl = self.explanations[i]

            pred_y = learner.predict(_densify(self.X[i]))[0]
            pred_expl = self.explain(learner, known_examples, i, pred_y)
            pred_expl = [(feat, int(np.sign(coeff)))
                         for feat, coeff in pred_expl.as_list()]

            matches = set(map(tuple, true_expl)).intersection(set(pred_expl))
            pr = len(matches) / len(pred_expl) if len(pred_expl) else 0.0
            rc = len(matches) / len(true_expl) if len(true_expl) else 0.0
            f1 = 0.0 if pr + rc <= 0 else 2 * pr * rc / (pr + rc)
            perfs.append((pr, rc, f1))

            if basename is None:
                continue

            self.save_expl(basename + '_{}_{}.txt'.format(i, t),
                           i, pred_y, pred_expl)
            self.save_expl(basename + '_{}_true.txt'.format(i),
                           i, true_y, true_expl)

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


class NewsgroupsProblem(TextProblem):
    def __init__(self, *args, classes=None, min_words=10, **kwargs):
        super().__init__(*args, **kwargs)

        path = join('data', '20newsgroups_{}_{}.pickle'.format(
                        '+'.join(sorted(classes)), min_words))
        try:
            dataset = load(path)
        except:
            raise RuntimeError('Run the data preparation script first!')

        self.class_names = classes
        self.y = dataset.target
        self.docs = dataset.data
        self.processed_docs = dataset.processed_data
        self.explanations = dataset.explanations

        self.vectorizer = TfidfVectorizer(lowercase=False).fit(self.processed_docs)
        self.X = self.vectorizer.transform(self.processed_docs)


class ReviewsProblem(TextProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        path = join('data', 'review_polarity_rationales.pickle')
        try:
            dataset = load(path)
        except:
            raise RuntimeError('Run the data preparation script first!')

        self.class_names = ['neg', 'pos']
        self.y = dataset['y']
        self.docs = self.processed_docs = dataset['docs']
        self.explanations = dataset['explanations']

        self.vectorizer = TfidfVectorizer(lowercase=False) \
                              .fit(self.processed_docs)
        self.X = self.vectorizer.transform(self.processed_docs)


class SVMLearner:
    def __init__(self, problem, strategy, C=1.0, kernel='linear', sparse=False,
                 rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        cv = StratifiedKFold(random_state=0)
        if not sparse:
            self._f_model = LinearSVC(C=C,
                                      penalty='l2',
                                      loss='hinge',
                                      multi_class='ovr',
                                      random_state=0)
        else:
            self._f_model = LinearSVC(C=C,
                                      penalty='l1',
                                      loss='squared_hinge',
                                      dual=False,
                                      multi_class='ovr',
                                      random_state=0)
        self._p_model = CalibratedClassifierCV(self._f_model,
                                               method='sigmoid',
                                               cv=cv)

        self.select_query = {
            'random': self._select_at_random,
            'least-confident': self._select_least_confident,
            'least-margin': self._select_least_margin,
        }[strategy]

    def select_model(self, X, y):
        Cs = np.logspace(-3, 3, 7)
        grid = GridSearchCV(estimator=self._f_model,
                            param_grid=dict(C=Cs),
                            scoring='f1_weighted',
                            n_jobs=-1)
        grid.fit(X, y)
        best_C = grid.best_estimator_.C
        print('SVM: setting C to', best_C)
        self._f_model.set_params(C=best_C)

    def _select_at_random(self, problem, examples):
        return self.rng.choice(sorted(examples))

    def _select_least_confident(self, problem, examples):
        examples = sorted(examples)
        margins = np.abs(self.decision_function(problem.X[examples]))
        # NOTE margins has shape (n_examples,) or (n_examples, n_classes)
        if margins.ndim == 2:
            margins = margins.min(axis=1)
        return examples[np.argmin(margins)]

    def _select_least_margin(self, problem, examples):
        examples = sorted(examples)
        probs = self.predict_proba(problem.X[examples])
        # NOTE probs has shape (n_examples, n_classes)
        diffs = np.zeros(probs.shape[0])
        for i, prob in enumerate(probs):
            sorted_indices = np.argsort(prob)
            diffs[i] = prob[sorted_indices[-1]] - prob[sorted_indices[-2]]
        return examples[np.argmin(diffs)]

    def fit(self, X, y):
        self._f_model.fit(X, y)
        self._p_model.fit(X, y)

    def get_params(self):
        return np.array(self._f_model.coef_, copy=True)

    def decision_function(self, X):
        return self._f_model.decision_function(X)

    def predict(self, X):
        return self._f_model.predict(X)

    def predict_proba(self, X):
        return self._p_model.predict_proba(X)


class LRLearner:
    def __init__(self, problem, strategy, C=1.0, kernel='linear', rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        self._model = LogisticRegression(C=C,
                                         penalty='l2',
                                         random_state=0)

        self.select_query = {
            'random': self._select_at_random,
            'least-confident': self._select_least_confident,
        }[strategy]

    def select_model(self, X, y):

        Cs = np.logspace(-3, 3, 7)
        grid = GridSearchCV(estimator=self._model,
                            param_grid=dict(C=Cs),
                            scoring='f1_weighted',
                            n_jobs=-1)
        grid.fit(X, y)
        best_C = grid.best_estimator_.C
        print('LR: setting C to', best_C)
        self._model.set_params(C=best_C)

    def _select_at_random(self, problem, examples):
        return self.rng.choice(sorted(examples))

    def _select_least_confident(self, problem, examples):
        examples = sorted(examples)
        margins = np.abs(self.decision_function(problem.X[examples]))
        # NOTE margins has shape (n_examples,) or (n_examples, n_classes)
        if margins.ndim == 2:
            margins = margins.min(axis=1)
        return examples[np.argmin(margins)]

    def fit(self, X, y):
        self._model.fit(X, y)

    def get_params(self):
        return np.array(self._model.coef_, copy=True)

    def decision_function(self, X):
        return self._model.decision_function(X)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class GPLearner:
    def __init__(self, problem, strategy, rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        self._r_model = GaussianProcessRegressor(random_state=0)
        self._c_model = GaussianProcessClassifier(random_state=0)

        self.select_query = {
            'random': self._select_at_random,
            'most-variance': self._select_most_variance,
        }[strategy]

    def _select_at_random(self, problem, examples):
        return self.rng.choice(sorted(examples))

    def _select_most_variance(self, problem, examples):
        examples = sorted(examples)
        _, std = self._r_model.predict(problem.X[examples], return_std=True)
        return examples[np.argmax(std)]

    def fit(self, X, y):
        self._r_model.fit(X, y)
        self._c_model.fit(X, y)

    def predict(self, X):
        return self._c_model.predict(X)

    def predict_proba(self, X):
        return self._c_model.predict_proba(X)


def caipi(problem,
          learner,
          train_examples,
          known_examples,
          test_examples,
          eval_examples,
          max_iters=100,
          start_expl_at=-1,
          eval_iters=10,
          improve_expl=False,
          basename=None,
          rng=None):
    rng = check_random_state(rng)

    print('CAIPI T={} #train={} #known={} #test={} #eval={}'.format(
          max_iters,
          len(train_examples), len(known_examples),
          len(test_examples), len(eval_examples)))

    X_test_tuples = {tuple(_densify(problem.X[i]).ravel()) for i in test_examples}

    learner.fit(problem.X[known_examples],
                problem.y[known_examples])

    perfs = []
    X_corr, y_corr = None, None
    for t in range(max_iters):

        if len(known_examples) >= len(train_examples):
            break

        unknown_examples = set(train_examples) - set(known_examples)
        i = learner.select_query(problem, unknown_examples)
        assert i in train_examples and i not in known_examples
        x = _densify(problem.X[i])

        explain = 0 <= start_expl_at <= t

        pred_y = learner.predict(x)[0]
        pred_expl = problem.explain(learner, known_examples, i, pred_y) \
                    if explain else None

        true_y = problem.query_label(i)
        known_examples.append(i)

        if explain and improve_expl:
            X_corr, y_corr = \
                problem.query_corrections(X_corr, y_corr, i, pred_y, pred_expl,
                                          X_test_tuples)
            raise NotImplementedError()

        learner.fit(vstack([X_corr, problem.X[known_examples]]),
                    hstack([y_corr, problem.y[known_examples]]))

        do_eval = t % eval_iters == 0
        perf = problem.eval(learner,
                            known_examples,
                            test_examples,
                            eval_examples if do_eval else None,
                            t=t, basename=basename)
        n_corrections = len(y_corr) if y_corr is not None else 0
        perf += (n_corrections,)

        # print('selecting model...')
        #if t >=5 and t % 5 == 0:
        #    learner.select_model(vstack([X_corr, problem.X[known_examples]]),
        #                         hstack([y_corr, problem.y[known_examples]]))

        params = np.round(learner.get_params(), decimals=1)
        print('{t:3d} : model = {params},  perfs = {perf}'.format(**locals()))
        perfs.append(perf)

    return perfs


def _subsample(problem, examples, prop, rng=None):
    rng = check_random_state(rng)

    classes = sorted(set(problem.y))
    n_sampled = int(round(len(examples) * prop))
    n_sampled_per_class = max(n_sampled // len(classes), 3)

    sample = []
    for y in classes:
        examples_y = np.array([i for i in examples if problem.y[i] == y])
        pi = rng.permutation(len(examples_y))
        sample.extend(examples_y[pi[:n_sampled_per_class]])

    return list(sample)


def eval_passive(problem, args, rng=None):
    """Useful for checking the based performance of the learner and whether
    the explanations are stable."""

    rng = check_random_state(rng)
    basename = _get_basename(args)

    folds = StratifiedShuffleSplit(n_splits=args.n_folds, random_state=rng) \
                .split(problem.y, problem.y)
    train_examples, test_examples = list(folds)[0]
    eval_examples = _subsample(problem, test_examples,
                               args.prop_eval, rng=rng)
    print('#train={} #test={} #eval={}'.format(
        len(train_examples), len(test_examples), len(eval_examples)))

    learner = LEARNERS[args.learner](problem, args.strategy, rng=0)
    learner.select_model(problem.X[train_examples],
                         problem.y[train_examples])
    learner.fit(problem.X[train_examples],
                problem.y[train_examples])
    train_params = learner.get_params()

    print('Computing full-train performance...')
    perf = problem.eval(learner, train_examples,
                        test_examples, eval_examples,
                        t='train', basename=basename)
    print('perf on full training set =', perf)

    print('Checking LIME stability...')
    perf = problem.eval(learner, train_examples,
                        test_examples, eval_examples,
                        t='train2', basename=basename)
    print('perf on full training set =', perf)

    print('Computing corrections for {} examples...'.format(len(train_examples)))
    X_test_tuples = {tuple(_densify(problem.X[i]).ravel())
                     for i in test_examples}

    X_corr, y_corr = None, None
    for j, i in enumerate(train_examples):
        print('  correcting {:3d} / {:3d}'.format(j + 1, len(train_examples)))
        x = _densify(problem.X[i])
        pred_y = learner.predict(x)[0]
        pred_expl = problem.explain(learner, train_examples, i, pred_y)
        X_corr, y_corr = problem.query_corrections(X_corr, y_corr, i, pred_y, pred_expl, X_test_tuples)

    if X_corr is None:
        print('no corrections were obtained')
        return
    print(X_corr.shape[0], 'corrections obtianed')

    print('Computing corr performance...')
    corr_params = None
    if np.min(y_corr) != np.max(y_corr):
        learner.select_model(X_corr, y_corr)
        learner.fit(X_corr, y_corr)
        corr_params = learner.get_params()
        perf = problem.eval(learner, train_examples,
                            test_examples, eval_examples,
                            t='corr', basename=basename)
    print('perf on corr only =', perf)

    print('Computing train+corr performance...')
    X_train_corr = vstack([problem.X[train_examples], X_corr])
    y_train_corr = hstack([problem.y[train_examples], y_corr])
    learner.select_model(X_train_corr, y_train_corr)
    learner.fit(X_train_corr, y_train_corr)
    train_corr_params = learner.get_params()
    perf = problem.eval(learner, train_examples,
                        test_examples, eval_examples,
                        t='train+corr', basename=basename)
    print('perf on train+corr set =', perf)

    print('w_train        :\n', train_params)
    print('w_corr         :\n', corr_params)
    print('w_{train+corr} :\n', train_corr_params)


def _eval_interactive(args, problem, rng=None):
    """The main evaluation loop."""

    rng = check_random_state(args.seed)
    basename = _get_basename(args)

    folds = StratifiedKFold(n_splits=args.n_folds, random_state=rng) \
                .split(problem.y, problem.y)

    perfs = []
    for k, (train_examples, test_examples) in enumerate(folds):
        print('Running fold {}/{}'.format(k + 1, args.n_folds))

        train_examples = list(train_examples)
        known_examples = _subsample(problem, train_examples,
                                    args.prop_known, rng=rng)
        test_examples = list(test_examples)
        eval_examples = _subsample(problem, test_examples,
                                   args.prop_eval, rng=rng)

        learner = LEARNERS[args.learner](problem, args.strategy, rng=0)

        perf = caipi(problem,
                     learner,
                     train_examples,
                     known_examples,
                     test_examples,
                     eval_examples,
                     max_iters=args.max_iters,
                     start_expl_at=args.start_expl_at,
                     eval_iters=args.eval_iters,
                     improve_expl=args.improve_expl,
                     basename=basename + '_fold={}'.format(k),
                     rng=rng)
        perfs.append(perf)

    dump(basename + '.pickle', {'args': args, 'perfs': perfs})


def _get_basename(args):
    fields = [
        ('problem', args.problem),
        ('learner', args.learner),
        ('strategy', args.strategy),
        ('n-folds', args.n_folds),
        ('prop-known', args.prop_known),
        ('prop-eval', args.prop_eval),
        ('max-iters', args.max_iters),
        ('start-expl-at', args.start_expl_at),
        ('eval-iters', args.eval_iters),
        ('improve-expl', args.improve_expl),
        ('n-features', args.n_features),
        ('n-samples', args.n_samples),
        ('kernel-width', args.kernel_width),
        ('seed', args.seed),
    ]
    basename = '__'.join([name + '=' + str(value) for name, value in fields])
    return join('results', basename)


PROBLEMS = {
    'toy-fst': lambda *args, **kwargs: \
            ToyProblem(*args, rule='fst', **kwargs),
    'toy-lst': lambda *args, **kwargs: \
            ToyProblem(*args, rule='lst', **kwargs),
    'colors-rule0': lambda *args, **kwargs: \
            ColorsProblem(*args, rule=0, **kwargs),
    'colors-rule1': lambda *args, **kwargs: \
            ColorsProblem(*args, rule=1, **kwargs),
    'ttt': TTTProblem,
    'newsgroups': lambda *args, **kwargs: \
            NewsgroupsProblem(*args,
                              classes=['sci.electronics', 'sci.med'],
                              **kwargs),
    'reviews': ReviewsProblem,
}


LEARNERS = {
    'svm': lambda *args, **kwargs: \
            SVMLearner(*args, sparse=False, **kwargs),
    'l1svm': lambda *args, **kwargs: \
            SVMLearner(*args, sparse=True, **kwargs),
    'lr': LRLearner,
    'gp': GPLearner,
}



def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('problem', choices=sorted(PROBLEMS.keys()),
                        help='name of the problem')
    parser.add_argument('learner', choices=sorted(LEARNERS.keys()),
                        default='svm', help='Active learner to use')
    parser.add_argument('strategy', type=str, default='random',
                        help='Query selection strategy to use')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')

    group = parser.add_argument_group('Evaluation')
    group.add_argument('-k', '--n-folds', type=int, default=10,
                       help='Number of cross-validation folds')
    group.add_argument('-p', '--prop-known', type=float, default=0.1,
                       help='Proportion of initial labelled examples')
    group.add_argument('-P', '--prop-eval', type=float, default=0.1,
                       help='Proportion of the test set to evaluate the '
                            'explanations on')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='Maximum number of learning iterations')
    group.add_argument('-e', '--eval-iters', type=int, default=10,
                       help='Interval for evaluating performance on the '
                       'evaluation set')
    group.add_argument('--passive', action='store_true',
                       help='DEBUG: eval perfs using passive learning')

    group = parser.add_argument_group('Interaction')
    group.add_argument('-E', '--start-expl-at', type=int, default=-1,
                       help='Iteration at which explanations kick in')
    group.add_argument('-I', '--improve-expl', action='store_true',
                       help='Whether the explanations should be improved')
    group.add_argument('-F', '--n-features', type=int, default=10,
                       help='Number of LIME features to present the user')
    group.add_argument('-S', '--n-samples', type=int, default=5000,
                       help='Size of the LIME sampled dataset')
    group.add_argument('-K', '--kernel-width', type=float, default=0.75,
                       help='LIME kernel width')
    args = parser.parse_args()

    np.seterr(all='raise')
    np.set_printoptions(precision=3, linewidth=80)

    rng = np.random.RandomState(args.seed)

    print('Creating problem...')
    problem = PROBLEMS[args.problem](args.n_samples,
                                     args.n_features,
                                     args.kernel_width,
                                     rng=rng)

    if args.passive:
        print('Evaluating passive learning...')
        eval_passive(problem, args, rng=rng)
    else:
        print('Evaluating interactive learning...')
        eval_interactive(problem, args, rng=rng)

if __name__ == '__main__':
    main()
