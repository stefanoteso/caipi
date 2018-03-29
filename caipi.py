#!/usr/bin/env python3

import numpy as np
import pickle
from sklearn.utils import check_random_state
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from itertools import product
from textwrap import dedent
from os.path import join
from blessings import Terminal


_TERM = Terminal()


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)


class PipeStep:
    def __init__(self, func):
        self.func = func
    def fit(self, *args, **kwargs):
        return self
    def transform(self, X):
        return self.func(X)


class TTTProblem:
    """Tic-tac-toe endgames.

    Classes: white wins (positive) vs. white does not win (negative)

    Interpretable features: 9 feature, one for each board slot; -1 means an
    o piece is at that position, 0 means that position is empty, 1 means an
    x piece is there.

    Uninterpretable features: binary, all three-position slots and possible
    places in those positions.
    """

    _PIECE_TO_INT = {'x': 1, 'b': 0, 'o': -1}

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

    def __init__(self, n_samples, n_features, min_coeff=1e-2, rng=None):
        self.n_samples = n_samples
        self.n_features = n_features
        self.min_coeff = min_coeff
        self.rng = check_random_state(rng)

        self.boards, y = [], []
        with open(join('data', 'tic-tac-toe.data'), 'rt') as fp:
            for line in map(str.strip, fp):
                chars = line.split(',')
                board = np.array([[chars[3*i+j] for j in range(3)]
                                  for i in range(3)], dtype=str)
                self.boards.append(board)
                y.append({'positive': 1, 'negative': 0}[chars[-1]])

        self.Z = np.array([self._board_to_z(board) for board in self.boards],
                          dtype=np.float64)
        self.X = np.array([self._z_to_x(z) for z in self.Z],
                          dtype=np.float64)
        self.y = np.array(y, dtype=np.int8)

        self.class_names = ('no-win', 'win')
        self.z_names = []
        for i, j in product(range(3), repeat=2):
            self.z_names.append('board[{i},{j}]'.format(**locals()))

    def _board_to_z(self, board):
        return np.array([self._PIECE_TO_INT[board[i, j]]
                         for i, j in product(range(3), repeat=2)],
                        dtype=np.float64)

    def _board_to_y(self, board):
        win = any([board[i, j] == ['x', 'x', 'x'] for i, j in self._TRIPLETS])
        return 1 if win else 0

    def _board_to_expl(self, board):
        SALIENT_CONFIGS = [
            # Win configurations
            (['x', 'x', 'x'], 1),
            # Almost-win configurations
            (['x', 'x', 'b'], -1),
            (['x', 'b', 'x'], -1),
            (['b', 'x', 'x'], -1),
            (['x', 'x', 'o'], -1),
            (['x', 'o', 'x'], -1),
            (['o', 'x', 'x'], -1),
        ]

        feats_weights = []
        for triplet in self._TRIPLETS:
            config = [board[i, j] for i, j in triplet]
            for salient_config, sign in SALIENT_CONFIGS:
                if config == salient_config:
                    for i, j in triplet:
                        piece = board[i, j]
                        if (piece == 'x' if sign else piece != 'x'):
                            s = self._PIECE_TO_INT[piece]
                            feat = 'board[{i},{j}]={s}'.format(**locals())
                            feats_weights.append((feat, sign))
        return feats_weights

    def _score_features(self, board, expl):
        scores = np.zeros((3, 3))
        for feat, coeff in expl:
            indices = feat.split('[')[-1].split(']')[0].split(',')
            value = int(feat.split('=')[-1])
            i, j = int(indices[0]), int(indices[1])
            if self._PIECE_TO_INT[board[i,j]] == value:
                scores[i, j] += coeff
        return scores

    @staticmethod
    def _z_to_x(z):
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
        """If the prediction is wrong, no feedback is provided."""
        true_y = self.y[i]
        if pred_y != true_y:
            return None, None

        """ Otherwise, the explanation may have to be corrected.  Since we do
        not show the user any false positives (in tic-tac-toe), all she can do
        is identify the *false positives*, if any.

        Considering feature relevance only, there is only one possible case:

        - prediction is correct, coefficient is wrongly non-zero:

          e.g. a winning board is classified as winning, but the highlighted
          pieces are not the winning triple of x's

          e.g. a losing board is classified as losing, but the highlighted
          pieces are not the o's blocking x's

        Therefore we duplicate the example by randomizing the highlighted
        pieces.

        With respect to polarity, there are two possible cases, but we don't
        care about them here.
        """
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

    def _get_pipeline(self, learner):
        step = PipeStep(lambda Z: np.array([self._z_to_x(z) for z in Z]))
        return make_pipeline(step, learner)

    def explain(self, learner, known_examples, i, y_pred):
        from lime.lime_tabular import LimeTabularExplainer
        from sklearn.linear_model import Ridge
        from time import time

        t = time()
        lime = LimeTabularExplainer(self.Z[known_examples],
                                    class_names=self.class_names,
                                    feature_names=self.z_names,
                                    categorical_features=list(range(self.Z.shape[1])),
                                    discretize_continuous=False,
                                    feature_selection='forward_selection',
                                    kernel_width=0.75,
                                    verbose=False)

        local_model = Ridge(alpha=1000, fit_intercept=True, random_state=0)
        pipeline = self._get_pipeline(learner)
        explanation = lime.explain_instance(self.Z[i],
                                            pipeline.predict_proba,
                                            model_regressor=local_model,
                                            num_samples=self.n_samples,
                                            num_features=self.n_features)
        print('LIME took', time() - t, 'seconds')
        return explanation

    def _eval_expl(self, true_z, pred_z):
        matches = set(true_z).intersection(set(pred_z))
        pr = len(matches) / len(pred_z) if len(pred_z) else 0.0
        rc = len(matches) / len(true_z) if len(true_z) else 0.0
        f1 = 2 * pr * rc / (pr + rc + 1e-12)
        return pr, rc, f1

    def eval(self, learner, known_examples, test_examples, eval_examples,
             t=None, basename=None):
        pred_perfs = prfs(self.y[test_examples],
                          learner.predict(self.X[test_examples]),
                          average='weighted')[:3]

        if eval_examples is None:
            expl_perfs = -1, -1, -1
        else:
            expl_perfs = []
            for i in eval_examples:
                true_y = self.y[i]
                true_z = self._board_to_expl(self.boards[i])

                pred_y = learner.predict(_densify(self.X[i]))[0]
                pred_z = [(feat, int(np.sign(coeff))) for feat, coeff in
                          self.explain(learner, known_examples, i, pred_y).as_list()]

                expl_perfs.append(self._eval_expl(true_z, pred_z))

                if basename is not None:
                    self.save_expl(basename + '_{}_{}.png'.format(i, t),
                                   i, pred_y, pred_z)
                    self.save_expl(basename + '_{}_true.png'.format(i),
                                   i, true_y, true_z)

            expl_perfs = np.mean(expl_perfs, axis=0)

        return tuple(pred_perfs) + tuple(expl_perfs)

    def save_expl(self, path, i, y, z):
        board = self.boards[i]
        scores = self._score_features(board, z)

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
            if board[i, j] != 'b':
                ax.plot(j + 0.5,
                        3 - (i + 0.5),
                        board[i, j],
                        markersize=25,
                        markerfacecolor=(0, 0, 0),
                        markeredgecolor=(0, 0, 0),
                        markeredgewidth=2)
            # Draw the highlight
            if np.abs(scores[i][j]) >= self.min_coeff:
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


class SVMLearner:
    def __init__(self, problem, strategy, C=1.0, kernel='linear', rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        cv = StratifiedKFold(random_state=0)
        self._f_model = LinearSVC(C=C,
                                  penalty='l2',
                                  loss='hinge',
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
        from sklearn.model_selection import GridSearchCV, cross_val_score

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

    def decision_function(self, X):
        return self._f_model.decision_function(X)

    def predict(self, X):
        return self._f_model.predict(X)

    def predict_proba(self, X):
        return self._p_model.predict_proba(X)


class GPLearner:
    def __init__(self, problem, strategy, rng=None):
        self.problem = problem
        self.rng = check_random_state(rng)

        from sklearn.gaussian_process import GaussianProcessRegressor, \
                                             GaussianProcessClassifier

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


def _stack(stack, arrays):
    arrays = [a for a in arrays if a is not None]
    if len(arrays) == 0:
        return None
    elif len(arrays) == 1:
        return arrays[0]
    return stack(arrays)


vstack = lambda arrays: _stack(np.vstack, arrays)
hstack = lambda arrays: _stack(np.hstack, arrays)


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

    learner.fit(problem.X[known_examples],
                problem.y[known_examples])

    perfs = []
    X_corr, y_corr = None, None
    for t in range(max_iters):

        if len(known_examples) >= len(train_examples):
            break

        # print('selecting a query instance...')
        unknown_examples = set(train_examples) - set(known_examples)
        i = learner.select_query(problem, unknown_examples)
        assert i in train_examples and i not in known_examples
        x = _densify(problem.X[i])

        explain = 0 <= start_expl_at <= t

        # print('predicting...')
        pred_y = learner.predict(x)[0]

        # print('explaining...')
        pred_z = problem.explain(learner, known_examples, i, pred_y) \
                 if explain else None

        # print('querying label...')
        true_y = problem.query_label(i)
        known_examples.append(i)

        if explain and improve_expl:
            # print('querying improved explanation...')
            X_extra, y_extra = problem.query_improved_expl(i, pred_y, pred_z)
            X_corr = vstack([X_corr, X_extra])
            y_corr = hstack([y_corr, y_extra])

        # print('fitting...')
        learner.fit(vstack([X_corr, problem.X[known_examples]]),
                    hstack([y_corr, problem.y[known_examples]]))

        # print('evaluating...')
        do_eval = t % eval_iters == 0
        perf = problem.eval(learner,
                            known_examples,
                            test_examples,
                            eval_examples if do_eval else None,
                            t=t, basename=basename)
        n_corrections = len(y_corr) if y_corr is not None else 0
        perf += (n_corrections,)

        # print('selecting model...')
        if t >=5 and t % 5 == 0:
            learner.select_model(vstack([X_corr, problem.X[known_examples]]),
                                 hstack([y_corr, problem.y[known_examples]]))

        print('iter {t:3d} : example #{i},  test perfs = {perf}'.format(**locals()))
        perfs.append(perf)

    return perfs


def get_basename(args):
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
        ('n-samples', args.n_samples),
        ('n-features', args.n_features),
        ('seed', args.seed),
    ]
    basename = '__'.join([name + '=' + str(value) for name, value in fields])
    return join('results', basename)


def subsample(problem, examples, prop, rng):
    classes = sorted(set(problem.y))
    n_sampled = max(round(len(examples) * prop), len(classes))
    n_sampled_per_class = max(n_sampled // len(classes), 3)

    sample = []
    for y in classes:
        y_examples = np.array([i for i in examples if problem.y[i] == y])
        pi = rng.permutation(len(y_examples))
        sample.extend(y_examples[pi[:n_sampled_per_class]])

    return list(sample)


PROBLEMS = {
    'ttt': TTTProblem,
}


LEARNERS = {
    'svm': SVMLearner,
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
                       help='Proportion of the test set to evaluate the'
                            'explanations on')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='Maximum number of learning iterations')
    group.add_argument('-e', '--eval-iters', type=int, default=10,
                       help='Interval for evaluating performance on the'
                       'evaluation set')
    group.add_argument('--eval-full', action='store_true',
                       help='DEBUG: eval perfs on training set beforehand')

    group = parser.add_argument_group('Interaction')
    group.add_argument('-E', '--start-expl-at', type=int, default=-1,
                       help='Iteration at which explanations kick in')
    group.add_argument('-I', '--improve-expl', action='store_true',
                       help='Whether the explanations should be improved')
    #group.add_argument('-f', '--freq-improvement', type=float, default=1,
    #                   help='Frequency with which improvements are made')
    #group.add_argument('-c', '--prop-improvement', type=float, default=1,
    #                   help='Proportion of explanation bits to be corrected')
    group.add_argument('-S', '--n-samples', type=int, default=5000,
                       help='Size of the LIME sampled dataset')
    group.add_argument('-F', '--n-features', type=int, default=10,
                       help='Number of LIME features to present the user')
    args = parser.parse_args()

    np.seterr(all='raise')
    np.set_printoptions(precision=3)
    rng = np.random.RandomState(args.seed)

    print('Creating problem...')
    problem = PROBLEMS[args.problem](args.n_samples, args.n_features, rng=rng)

    basename = get_basename(args)

    folds = StratifiedKFold(n_splits=args.n_folds, random_state=rng) \
                .split(problem.y, problem.y)

    perfs = []
    for k, (train_examples, test_examples) in enumerate(folds):
        print('Running fold {}/{}'.format(k + 1, args.n_folds))

        train_examples = list(train_examples)
        known_examples = subsample(problem, train_examples,
                                   args.prop_known, rng)
        test_examples = list(test_examples)
        eval_examples = subsample(problem, test_examples,
                                  args.prop_eval, rng)

        learner = LEARNERS[args.learner](problem, args.strategy, rng=0)

        if args.eval_full:

            print('Computing full-train performance...')
            learner.fit(problem.X[train_examples],
                        problem.y[train_examples])
            perf = problem.eval(learner, train_examples,
                                test_examples, eval_examples)
            print('perf on full training set =', perf)

            print('Computing augmented-train performance...')
            X_corr, y_corr = None, None
            for i in train_examples:
                x = _densify(problem.X[i])
                pred_y = learner.predict(x)[0]
                pred_z = problem.explain(learner, train_examples, i, pred_y)
                X_extra, y_extra = \
                    problem.query_improved_expl(i, pred_y, pred_z)
                X_corr = vstack([X_corr, X_extra])
                y_corr = hstack([y_corr, y_extra])
            learner.fit(vstack([X_corr, problem.X[train_examples]]),
                        hstack([y_corr, problem.y[train_examples]]))
            perf = problem.eval(learner, train_examples,
                                test_examples, eval_examples)
            print('perf on augmented training set =', perf)

            quit()

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
                     basename=basename,
                     rng=rng)
        perfs.append(perf)

    dump(get_basename(args) + '.pickle', {'args': args, 'perfs': perfs})


if __name__ == '__main__':
    main()
