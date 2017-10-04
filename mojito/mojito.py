import numpy as np
from sklearn.utils import check_random_state
from textwrap import dedent

from .utils import TextMod


def mojito(problem, learner, explainer, train_examples, known_examples,
           max_iters=100, start_explaining_at=20, rng=None):
    """An implementation of the Mojito algorithm.

    Parameters
    ----------
    problem : mojito.Problem
        The problem.
    learner : mojito.ActiveLearner
        The learner.
    explainer : mojito.Explainer
        The explainer.
    train_examples : list of int
        Indices of the training examples
    known_examples : list of int
        Indices of the examples whose label is known.
    max_iters : int, defaults to 100
        Maximum number of iterations.
    start_explaining_at : int, default to 20
        Iteration at which the explanation mechanic kicks in.
    rng : numpy.RandomState, defaults to None
        The RNG.
    """
    rng = check_random_state(rng)

    train_examples = list(train_examples)
    known_examples = list(known_examples)
    test_examples = list(set(problem.examples) - set(train_examples))

    # Fit a model on the complete training set
    learner.fit(problem.X[train_examples], problem.Y[train_examples])
    full_perfs = problem.evaluate(learner,
                                  problem.X[test_examples],
                                  problem.Y[test_examples])

    # Fit an initial model on the known examples
    learner.fit(problem.X[known_examples], problem.Y[known_examples])
    trace = [problem.evaluate(learner,
                              problem.X[test_examples],
                              problem.Y[test_examples])]

    perfs = trace[-1]
    print(dedent('''\
            T={} #train={} #known={} #test={}
            full set perfs = {}
            starting perfs = {}
        ''').format(max_iters, len(train_examples), len(known_examples),
                    len(test_examples), full_perfs, perfs))

    explain = False
    for t in range(max_iters):
        if len(known_examples) == len(train_examples):
            break
        if t >= start_explaining_at:
            explain = True

        # Select a query from the unknown examples
        i = learner.select_query(problem.X, problem.Y,
                                 set(train_examples) - set(known_examples))
        assert i in train_examples and i not in known_examples

        # Compute a prediction and an explanation
        x_explainable = problem.X_explainable[i]
        y = learner.predict(problem.X[i].reshape(1, -1))

        g = None if not explain else \
            explainer.explain(problem, learner, problem.X_explainable[i])

        # Ask the user
        y_bar, g_bar = problem.improve(i, y, g)

        # Update the model
        known_examples.append(i)
        learner.fit(problem.X[known_examples],
                    problem.Y[known_examples])

        # Record the model performance
        perfs = problem.evaluate(learner,
                                 problem.X[test_examples],
                                 problem.Y[test_examples])
        print('{t:3d} : example {i}, label {y} -> {y_bar}, perfs {perfs}'
                  .format(**locals()))
        trace.append(np.array(perfs))
    else:
        print('all examples processed in {} iterations'.format(t))

    return np.array(trace)
