import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as prfs


class Evaluator:
    """Evaluates the quality of the predictions and explanations.

    Predictions are evaluated in terms of precision, recall and F1 over the
    test set.

    Explanations are evaluated based on how close they are to the explanations
    given by a pseudo-oracle, namely a learner fit on the entire dataset
    (including the test set).
    """
    def __init__(self, problem, oracle_kind='l1logreg',
                 num_samples=5000, num_features=10):
        self.problem = problem
        if oracle_kind == 'l1logreg':
            oracle = LogisticRegression(penalty='l1', C=1, max_iter=10,
                                        random_state=0)
        elif oracle_kind == 'tree':
            oracle = DecisionTreeClassifier(random_state=0)
        else:
            raise ValueError('unsupported oracle_kind={}'.format(oracle_kind))

        self.oracle = problem.wrap_preproc(oracle).fit(problem.X, problem.Y)
        self.num_samples = num_samples
        self.num_features = num_features

    def evaluate_predictions(self, learner, examples):
        """Evaluates a learner's predictions.

        The learner must be already wrapped in the preprocessing pipeline,
        if any.
        """
        X_examples = self.problem.X[examples]
        Y_hat = learner.predict(X_examples)
        return prfs(self.problem.Y[examples], Y_hat, average='weighted')[:3]

    def evaluate_explanation(self, learner, explanation, example, y):
        """Computes the recall over the true features."""
        if explanation is None:
            return -1, -1

        # TODO the oracle is interpretable directly, we should not use LIME
        assert example is not None and y is not None
        oracle_explanation = \
            self.problem.explain(self.oracle,
                                 self.problem.examples,
                                 example, y,
                                 num_samples=self.num_samples,
                                 num_features=self.num_features)
        perf = self.problem.get_explanation_perf(oracle_explanation,
                                                 explanation)
        return perf, explanation.score

    def evaluate(self, learner, examples, explanation=None, example=None, y=None):
        """Evaluates predictions an explanations."""
        return self.evaluate_predictions(learner, examples) + \
               self.evaluate_explanation(learner, explanation, example, y)
