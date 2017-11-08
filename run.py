#!/usr/bin/env python3

import argparse
import numpy as np
import mojito

from os.path import join
from pprint import pprint
from sklearn.model_selection import StratifiedKFold


PROBLEMS = {
    'cancer': mojito.CancerProblem,
    'newsgroups': mojito.NewsgroupsProblem,
    'sport': lambda *args, **kwargs: \
        mojito.NewsgroupsProblem(*args,
            labels=('rec.sport.baseball', 'rec.sport.hockey'), **kwargs),
    'religion': lambda *args, **kwargs: \
        mojito.NewsgroupsProblem(*args,
            labels=('alt.atheism', 'soc.religion.christian'), **kwargs),
    'character': mojito.CharacterProblem,
}


LEARNERS = {
    'svm': mojito.ActiveSVM,
    'gp': mojito.ActiveGP,
}


def get_results_path(args):
    fields = [
        ('learner', args.learner),
        ('strategy', args.strategy),
        ('num-folds', args.num_folds),
        ('perc-known', args.perc_known),
        ('max-iters', args.max_iters),
        ('start-explaining-at', args.start_explaining_at),
        ('num-samples', args.num_samples),
        ('num-features', args.num_features),
        ('improve-explanations', args.improve_explanations),
        ('seed', args.seed),
    ]
    filename = 'traces_{}_'.format(args.problem) + \
               '_'.join([name + '=' + str(value) for name, value in fields]) + \
               '.pickle'
    return join('results', filename)


def fit_oracle(problem, rng):
    """Fits an 'oracle' learner to the dataset using interpretable features."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty='l1', C=1)
    return model.fit(problem.X, problem.Y)


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('problem', help='name of the problem')
    parser.add_argument('learner', type=str, default='svm',
                        help='Active learner to use')
    parser.add_argument('strategy', type=str, default='random',
                        help='Query selection strategy to use')
    parser.add_argument('-f', '--num-folds', type=int, default=10,
                        help='Number of cross-validation folds')
    parser.add_argument('-p', '--perc-known', type=float, default=10,
                        help='Percentage of initial labelled examples')
    parser.add_argument('-T', '--max-iters', type=int, default=100,
                        help='Maximum number of learning iterations')
    parser.add_argument('-E', '--start-explaining-at', type=int, default=-1,
                        help='Iteration at which explanations kick in')
    parser.add_argument('-k', '--num-samples', type=int, default=5000,
                        help='Size of the LIME sampled dataset')
    parser.add_argument('-n', '--num-features', type=int, default=10,
                        help='Number of LIME features to present the user')
    parser.add_argument('-e', '--improve-explanations', action='store_true',
                        help='Whether the explanations should be improved')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')
    args = parser.parse_args()

    np.seterr(all='raise')
    np.set_printoptions(precision=3)
    rng = np.random.RandomState(args.seed)

    print('Creating problem...')
    problem = PROBLEMS[args.problem](rng=rng)
    folds = StratifiedKFold(n_splits=args.num_folds, random_state=rng) \
                .split(problem.Y, problem.Y)

    # Fit an interpretable model on the full dataset
    oracle = fit_oracle(problem, rng)

    traces = []
    for k, (train_examples, test_examples) in enumerate(folds):
        print('Running fold {}/{}'.format(k + 1, args.num_folds))

        learner = LEARNERS[args.learner](args.strategy, rng=rng)

        num_known = max(round(len(train_examples) * (args.perc_known / 100)), 2)
        pi = rng.permutation(len(train_examples))
        known_examples = train_examples[pi[:num_known]]

        traces.append(mojito.mojito(problem, learner,
                                    train_examples, known_examples, oracle,
                                    max_iters=args.max_iters,
                                    start_explaining_at=args.start_explaining_at,
                                    improve_explanations=args.improve_explanations,
                                    num_samples=args.num_samples,
                                    num_features=args.num_features,
                                    rng=rng))

    mojito.dump(get_results_path(args),
                {'args': args, 'num_examples': len(problem.examples),
                 'traces': traces})


if __name__ == '__main__':
    main()
