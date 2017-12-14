#!/usr/bin/env python3

import datetime
import argparse
import numpy as np
import mojito


from os.path import join
from pprint import pprint
from sklearn.model_selection import StratifiedKFold


PROBLEMS = {
    'iris': mojito.IrisProblem,
    'cancer': mojito.CancerProblem,
    'newsgroups': mojito.NewsgroupsProblem,
    'newsgroups-sport': lambda *args, **kwargs:
    mojito.NewsgroupsProblem(*args,
                             labels=('rec.sport.baseball', 'rec.sport.hockey'), **kwargs),
    'newsgroups-religion': lambda *args, **kwargs:
    mojito.NewsgroupsProblem(*args,
                             labels=('alt.atheism', 'soc.religion.christian'), **kwargs),
    'mnist-binary': lambda *args, **kwargs:
    mojito.MNISTProblem(*args, labels=(5, 6), **kwargs),
    'mnist-multiclass': mojito.MNISTProblem,
    'fer13-binary': lambda *args, **kwargs:
    mojito.FER13Problem(*args, labels=(2, 5), **kwargs),
    'fer13-multiclass': mojito.FER13Problem,
}


LEARNERS = {
    'svm': mojito.ActiveSVM,
    'gp': mojito.ActiveGP,
}


def get_results_path(args):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fields = [
        ('learner', args.learner),
        ('strategy', args.strategy),
        ('num-folds', args.num_folds),
        ('perc-known', args.perc_known),
        ('max-iters', args.max_iters),
        ('start-explaining-at', args.start_explaining_at),
        ('num-samples', args.num_samples),
        ('num-features', args.num_features),
        ('eval-explanations-every', args.eval_explanations_every),
        ('improve-explanations', args.improve_explanations),
        ('seed', args.seed),
    ]
    filename = 'traces_{}_{}_'.format(date_str, args.problem) + \
               '_'.join([name + '=' + str(value) for name, value in fields]) + \
               '.pickle'
    return join('results', filename)


def sample_examples(problem, train_examples, perc_known, rng):
    """Samples a subset of examples, ensures that each class is represented."""
    classes = sorted(set(problem.y))
    num_known = max(round(len(train_examples) * (perc_known / 100)),
                    len(classes))
    num_known_per_class = max(num_known // len(classes), 3)
    selected = []
    for y in classes:
        y_examples = np.array([i for i in train_examples if problem.y[i] == y])
        pi = rng.permutation(len(y_examples))
        selected.extend(y_examples[pi[:num_known_per_class]])
    return np.array(selected)


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('problem', choices=sorted(PROBLEMS.keys()),
                        help='name of the problem')
    parser.add_argument('learner', type=str, default='svm',
                        help='Active learner to use')
    parser.add_argument('strategy', type=str, default='random',
                        help='Query selection strategy to use')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')
    parser.add_argument('--learner-args', type=str, nargs='?',
                        help='Additional sklearn parameters for the learner' +
                        ' in the form of a list attribute=val' +
                        ' "name1=val1,..,namek=valk"')

    group = parser.add_argument_group('Explanations')
    group.add_argument('-E', '--start-explaining-at', type=int, default=-1,
                       help='Iteration at which explanations kick in')
    group.add_argument('-I', '--improve-explanations', action='store_true',
                       help='Whether the explanations should be improved')
    group.add_argument('-S', '--num-samples', type=int, default=5000,
                       help='Size of the LIME sampled dataset')
    group.add_argument('-F', '--num-features', type=int, default=10,
                       help='Number of LIME features to present the user')
    group.add_argument('-e', '--eval-explanations-every', type=int, default=10,
                       help='Interval for evaluating explanation performance'
                            'on the test set')
    group.add_argument('-d', '--discretize', action='store_true',
                       help='Whether the to discretize continuous features on tabular data')

    group = parser.add_argument_group('Evaluation')
    group.add_argument('-k', '--num-folds', type=int, default=10,
                       help='Number of cross-validation folds')
    group.add_argument('-p', '--perc-known', type=float, default=10,
                       help='Percentage of initial labelled examples')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='Maximum number of learning iterations')
    group.add_argument('-O', '--oracle-kind', type=str, default='l1logreg',
                       help='Kind of explanation oracle to use')
    group.add_argument('-x', '--n-test-examples-exp', type=int, default=20,
                       help='Number of test examples to use for measuring explanation perfs')
    args = parser.parse_args()

    np.seterr(all='raise')
    np.set_printoptions(precision=3)
    rng = np.random.RandomState(args.seed)

    print('Creating problem...')
    problem = PROBLEMS[args.problem](rng=rng)
    folds = StratifiedKFold(n_splits=args.num_folds, random_state=rng) \
        .split(problem.y, problem.y)

    learner_args = None
    if args.learner_args is not None:
        learner_key_value_pairs = args.learner_args.split(',')
        learner_args = {key.strip(): value.strip() for key, value in
                        [pair.strip().split('=')
                         for pair in learner_key_value_pairs]}
    else:
        learner_args = {}
    print('Learner additional parameters {}'.format(learner_args))

    print('Fitting the {} oracle...'.format(args.oracle_kind))
    # evaluator = mojito.Evaluator(problem,
    #                              oracle_kind=args.oracle_kind,
    #                              num_samples=args.num_samples,
    #                              num_features=args.num_features)
    # oracle_perfs = evaluator.evaluate(evaluator.oracle, problem.examples)
    # print('oracle perfs = {}'.format(oracle_perfs))

    traces, explanation_perfs = [], []
    for k, (train_examples, test_examples) in enumerate(folds):
        print('Running fold {}/{}'.format(k + 1, args.num_folds))

        evaluator = mojito.Evaluator(problem,
                                     oracle_kind=args.oracle_kind,
                                     num_samples=args.num_samples,
                                     num_features=args.num_features,
                                     train_examples=train_examples)
        oracle_perfs = evaluator.evaluate(evaluator.oracle, problem.examples,
                                          discretize=args.discretize)
        print('oracle perfs = {}'.format(oracle_perfs))

        learner = LEARNERS[args.learner](problem, args.strategy, rng=rng, **learner_args)
        known_examples = sample_examples(problem, train_examples,
                                         args.perc_known, rng)

        trace, explanation_perf = \
            mojito.mojito(problem, evaluator, learner,
                          train_examples, known_examples,
                          max_iters=args.max_iters,
                          start_explaining_at=args.start_explaining_at,
                          improve_explanations=args.improve_explanations,
                          num_samples=args.num_samples,
                          num_features=args.num_features,
                          eval_explanations_every=args.eval_explanations_every,
                          discretize=args.discretize,
                          n_test_examples_exp=args.n_test_examples_exp,
                          rng=rng)

        traces.append(trace)
        explanation_perfs.append(explanation_perf),

    mojito.dump(get_results_path(args), {
        'args': args,
        'num_examples': len(problem.examples),
        'traces': traces,
        'explanation_perfs': explanation_perfs,
    })


if __name__ == '__main__':
    main()
