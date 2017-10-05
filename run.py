#!/usr/bin/env python3

import argparse
import numpy as np
import mojito

from os.path import join
from pprint import pprint
from sklearn.model_selection import KFold


PROBLEMS = {
    'cancer': mojito.CancerProblem,
    'religion': mojito.ReligionProblem,
}


def get_traces_path(args):
    temp = 'traces_{}_seed={}.pickle'.format(args.problem, args.seed)
    return join('results', temp)


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('problem', help='name of the problem')
    parser.add_argument('-S', '--strategy', type=str, default='random',
                        help='Query selection strategy to use')
    parser.add_argument('-T', '--max-iters', type=int, default=100,
                        help='Maximum number of learning iterations')
    parser.add_argument('-k', '--num-folds', type=int, default=10,
                        help='Number of cross-validation folds')
    parser.add_argument('-p', '--perc-known', type=float, default=10,
                        help='Percentage of initial labelled examples')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')
    args = parser.parse_args()

    np.seterr(all='raise')
    np.set_printoptions(precision=3)
    rng = np.random.RandomState(args.seed)

    print('Creating problem...')
    oracle = mojito.ActiveSVM(rng=rng)
    problem = PROBLEMS[args.problem](oracle=oracle, rng=rng)
    folds = KFold(n_splits=args.num_folds, random_state=rng) \
                .split(problem.examples)

    traces = []
    for k, (train_examples, test_examples) in enumerate(folds):
        print('Running fold {}/{}'.format(k + 1, args.num_folds))

        learner = mojito.ActiveSVM(args.strategy, rng=rng)
        explainer = mojito.LimeExplainer(problem, rng=rng)

        num_known = max(round(len(train_examples) * (args.perc_known / 100)), 1)
        pi = rng.permutation(len(train_examples))
        known_examples = train_examples[pi[:num_known]]

        traces.append(mojito.mojito(problem, learner, explainer,
                                    train_examples, known_examples,
                                    max_iters=args.max_iters,
                                    rng=rng))
        print('Results:')
        pprint(traces[-1])
        quit()

    mojito.dump(get_traces_path(args), traces)


if __name__ == '__main__':
    main()
