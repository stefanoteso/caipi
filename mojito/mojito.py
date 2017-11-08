import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import precision_recall_fscore_support as prfs
from textwrap import dedent

from .utils import TextMod, densify


def as_coeff(g):
    return np.array([coeff for _, coeff in g.as_list()]) if g else None


def dist(x, z):
    return np.linalg.norm(x - z)


def evaluate(problem, learner, examples, explanations=None):
    """Computes precision, recall and F1 of a prediction."""
    X, Y = problem.X[examples], problem.Y[examples]
    a = prfs(Y, learner.predict(X), average='weighted')[:3]
    b = None
    if explanations is not None:
        # TODO compute fraction of gold features that are recovered by LIME
        b = dist(as_coeff(explanations[0]), as_coeff(explanations[1]))
    return a + (b,)


def mojito(problem, learner, train_examples, known_examples, oracle,
           max_iters=100, start_explaining_at=-1, improve_explanations=False,
           num_samples=5000, num_features=10, rng=None):
    """An implementation of the Mojito algorithm.

    Parameters
    ----------
    problem : mojito.Problem
        The problem.
    learner : mojito.ActiveLearner
        The learner.
    train_examples : list of int
        Indices of the training examples
    known_examples : list of int
        Indices of the examples whose label is known.
    oracle : mojito.ActiveLearner
        A miracle know-it-all learner.
    max_iters : int, defaults to 100
        Maximum number of iterations.
    start_explaining_at : int, default to -1
        Iteration at which the explanation mechanic kicks in. Negative means
        disabled.
    improve_explanations : bool, defaults to True
        Whether to obtain feedback on the explanations.
    num_samples : int, defaults to 5000
        Number of samples used by LIME.
    num_features : int, defaults to 10
        Number of explanatory features used by LIME
    rng : numpy.RandomState, defaults to None
        The RNG.
    """
    rng = check_random_state(rng)

    train_examples = list(train_examples)
    known_examples = list(known_examples)
    test_examples = list(set(problem.examples) - set(train_examples))

    # Fit a model on the complete training set
    # learner.fit(problem.X[train_examples], problem.Y[train_examples])
    full_perfs = (0, 0, 0) # problem.evaluate(learner, test_examples)

    # Fit an initial model on the known examples
    learner.fit(problem.X[known_examples], problem.Y[known_examples])
    trace = [evaluate(problem, learner, test_examples) + (-1, -1, -1)]

    print(dedent('''\
            T={} #train={} #known={} #test={}
            full set perfs = {}
            starting perfs = {}
        ''').format(max_iters, len(train_examples), len(known_examples),
                    len(test_examples), full_perfs, trace[-1]))

    num_errors, explain = 0, False
    for t in range(max_iters):
        if len(known_examples) == len(train_examples):
            break
        if 0 <= start_explaining_at <= t:
            explain = True

        # Select a query from the unknown examples
        unknown_examples = set(train_examples) - set(known_examples)
        i = learner.select_query(problem.X, problem.Y, unknown_examples)
        assert i in train_examples and i not in known_examples

        # Compute the prediction
        x = densify(problem.X[i])
        y = learner.predict(x.reshape(1, -1))[0]

        # Compute the learner's explanation
        g = (problem.explain(learner, known_examples, i,
                             num_samples=num_samples,
                             num_features=num_features)
             if explain else None)

        # Compute the 'gold' explanation
        g_star = (problem.explain(oracle, problem.examples, i,
                                  num_samples=num_samples)
                  if explain else None)

        # Ask the true label to the user
        y_bar = problem.improve(i, y)
        known_examples.append(i)

        num_errors += 1 if y != y_bar else 0

        # Ask an improved explanation to the user
        g_bar = (problem.improve_explanation(i, y, g)
                 if explain and improve_explanations else None)

        # Update the model
        learner.fit(problem.X[known_examples], problem.Y[known_examples])

        # TODO learn from explanation improvements

        # Record the model performance
        y_diff = y_bar - y
        perfs = evaluate(problem, learner, test_examples,
                         explanations=(g, g_star) if explain else None)

        # TODO count iterations with explanation improvements but no error

        v, v_bar, v_star = as_coeff(g), as_coeff(g_bar), as_coeff(g_star)
        print(dedent('''\
                iter {t:3d} : #{i}  label change {y_diff}  perfs {perfs}
                explanations:
                model  = {v}
                user   = {v_bar}
                oracle = {v_star}
                '''.format(**locals())))

        trace.append(perfs + (num_errors, g, g_bar, g_star))

    else:
        print('all examples processed in {} iterations'.format(t))

    return np.array(trace)
