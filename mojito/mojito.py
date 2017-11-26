import numpy as np
from sklearn.utils import check_random_state
from textwrap import dedent

from .utils import densify


def _densify_one(x):
    """Converts an example to a dense ndarray of shape (1, n_features)."""
    x = densify(x)
    if x.shape[0] != 1:
        # NOTE if X[i] is already dense, densify(X[i]) is a no-op; in this
        # case we get an x of shape (n_features,), and we turn it into (1,
        # n_features); if X[i] is sparse, densify(X[i]) returns an x of
        # shape (1, n_features), so we don't have to "unravel" it.
        x = x[np.newaxis, ...]
    return x


def predict_and_explain(problem, learner, known_examples, example,
                        explain=False, num_samples=5000, num_features=10):
    # Compute the prediction
    y = learner.predict(_densify_one(problem.X[example]))[0]

    # Compute the explanation
    g = (problem.explain(learner, known_examples, example, y,
                         num_samples=num_samples,
                         num_features=num_features)
         if explain else None)

    return y, g


def mojito(problem, evaluator, learner, train_examples, known_examples,
           max_iters=100, start_explaining_at=-1, improve_explanations=False,
           num_samples=5000, num_features=10, eval_explanations_every=10,
           rng=None):
    """An implementation of the Mojito algorithm.

    Parameters
    ----------
    problem : mojito.Problem
        The problem.
    learner : mojito.ActiveLearner
        The learner.
    evalutor : mojito.Evaluator
        The performance evaluator.
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
    num_samples : int, defaults to 5000
        Number of samples used by LIME.
    num_features : int, defaults to 10
        Number of explanatory features used by LIME
    eval_explanations_every : int, defaults to 10
        Interval (in iterations) between explanation evaluations.
    """
    rng = check_random_state(rng)

    train_examples = list(train_examples)
    known_examples = list(known_examples)
    test_examples = list(set(problem.examples) - set(train_examples))
    expl_test_examples = rng.permutation(test_examples)[:20]

    # Wrap the learner in the preprocessing pipeline, if any
    unwrapped_learner = learner
    learner = problem.wrap_preproc(learner)

    # Fit the model on the complete training set (for debug only)
    learner.fit(problem.X[train_examples], problem.y[train_examples])
    full_perfs = evaluator.evaluate(learner, test_examples)

    # Fit the initial model on the known examples
    learner.fit(problem.X[known_examples], problem.y[known_examples])
    initial_perfs = evaluator.evaluate(learner, test_examples)

    print(dedent('''\
            T={} #train={} #known={} #test={}
            full training set perfs = {}
            initial perfs = {}
        ''').format(max_iters, len(train_examples), len(known_examples),
                    len(test_examples), full_perfs, initial_perfs))

    num_errors = 0
    trace, explanation_perfs = [initial_perfs + (0,)], []
    for t in range(max_iters):

        if len(known_examples) >= len(train_examples):
            break

        # TODO learn from explanation improvements
        # TODO count iterations with explanation improvements but no error

        # Select a query from the unknown examples
        i = unwrapped_learner.select_query(problem,
                set(train_examples) - set(known_examples))

        # Predict and explain
        explain = 0 <= start_explaining_at <= t
        y, g = predict_and_explain(problem, learner, known_examples, i,
                                   explain=explain, num_samples=num_samples,
                                   num_features=num_features)

        # Ask the true label to the user
        y_bar = problem.improve(i, y)
        known_examples.append(i)

        # Ask an improved explanation to the user
        g_bar = (problem.improve_explanation(i, y, g)
                 if explain and improve_explanations else None)

        # Update the model
        learner.fit(problem.X[known_examples], problem.y[known_examples])

        # Record the model performance
        y_diff = y_bar != y
        num_errors += y_diff
        perfs = evaluator.evaluate(learner, test_examples,
                                   example=i, y=y, explanation=g)
        print('iter {t:3d} : #{i}  y changed? {y_diff}  perfs={perfs}'
                  .format(**locals()))
        trace.append(perfs + (num_errors,))

        # Evaluate explanation performance on the test set
        if not explain:
            continue

        if t % eval_explanations_every == 0:
            print('evaluating explanation performance...')
            perfs = []
            for i, example in enumerate(expl_test_examples):
                y, g = predict_and_explain(problem, learner,
                                           known_examples[:-1], example,
                                           explain=True,
                                           num_samples=num_samples,
                                           num_features=num_features)
                perf = evaluator.evaluate_explanation(example, y, g)
                print(' {}/{} : {}'.format(i, len(expl_test_examples), perf))
                perfs.append(perf)
            explanation_perfs.append(perfs)

    return np.array(trace), np.array(explanation_perfs)
