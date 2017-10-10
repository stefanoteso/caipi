import numpy as np
from sklearn.utils import check_random_state
from textwrap import dedent

from .utils import TextMod


def mojito(problem, learner, explainer, train_examples, known_examples,
           max_iters=100, start_explaining_at=-1, improve_explanations=False,
           rng=None):
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
    start_explaining_at : int, default to -1
        Iteration at which the explanation mechanic kicks in. Negative means
        disabled.
    improve_explanations : bool, defaults to True
        Whether to obtain feedback on the explanations.
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
                              problem.Y[test_examples]) + (-1, -1)]

    print(dedent('''\
            T={} #train={} #known={} #test={}
            full set perfs = {}
            starting perfs = {}
        ''').format(max_iters, len(train_examples), len(known_examples),
                    len(test_examples), full_perfs, trace[-1]))

    explain = False
    for t in range(max_iters):
        if len(known_examples) == len(train_examples):
            break
        if 0 <= start_explaining_at <= t:
            explain = True

        # Select a query from the unknown examples
        i = learner.select_query(problem.X, problem.Y,
                                 set(train_examples) - set(known_examples))
        assert i in train_examples and i not in known_examples

        # Compute a prediction and an explanation
        x = problem.X[i]
        y = learner.predict(x.reshape(1, -1))[0]

        x_explainable = problem.X_explainable[i]
        if explain:
            g, v, c, discrepancy, X_sampled, Z_sampled = \
                explainer.explain(problem, learner, x_explainable)
        else:
            g, v, c, discrepancy, X_sampled, Z_sampled = \
                None, None, None, -1, None, None

        # Ask the user
        y_bar = problem.improve(i, y)
        if explain and improve_explanations:
            g_bar, v_bar, c_bar, discrepancy_bar, _, _ = \
                problem.improve_explanation(explainer, x_explainable, y, g)
        else:
            g_bar, v_bar, c_bar, discrepancy_bar = None, None, None, -1

        # Debug
        if g is not None:
            print('model explanation (discrepancy {discrepancy}) =\n {g}'
                     .format(**locals()))
        if g_bar is not None:
            print('oracle explanation (discrepancy {discrepancy_bar}) =\n {g_bar}'
                     .format(**locals()))

        # Update the model
        known_examples.append(i)
        if explain and improve_explanations and g is not None and g_bar is not None:
            R_sampled = explainer.rank_labels(Z_sampled, g, c, g_bar, c_bar)
            learner.fit(problem.X[known_examples], problem.Y[known_examples],
                        X_lime=X_sampled, R_lime=R_sampled)
        else:
            learner.fit(problem.X[known_examples], problem.Y[known_examples])

        # TODO: learn from the improved explanation

        # Record the model performance
        y_diff = y_bar - y
        perfs = problem.evaluate(learner,
                                 problem.X[test_examples],
                                 problem.Y[test_examples])
        print('{t:3d} : example {i}, label change {y_diff}, perfs {perfs}'
                  .format(**locals()))

        trace.append(perfs + (discrepancy, discrepancy_bar))
    else:
        print('all examples processed in {} iterations'.format(t))

    return np.array(trace)
