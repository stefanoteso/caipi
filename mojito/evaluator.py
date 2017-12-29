import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from itertools import product


class Oracle:
    """Evaluates the quality of the predictions and explanations.

    Predictions are evaluated in terms of precision, recall and F1 over the
    test set.

    Explanations are evaluated in terms of explanation precision, recall and
    F1 over the generated explanation.
    """
    def __init__(self, problem):
        self.problem = problem

    def evaluate_predictions(self, learner, examples):
        """Evaluates the learner's prediction."""
        return prfs(self.problem.y[examples],
                    learner.predict(self.problem.X[examples]),
                    average='weighted')[:3]

    def evaluate_explanation(self, example, y, explanation):
        """Computes the learner's explanation."""
        raise NotImplementedError()

    def evaluate(self, learner, test_examples, explanation=None, example=None, y=None):
        """Evaluates predictions an explanations."""
        return self.evaluate_predictions(learner, test_examples) + \
               self.evaluate_explanation(example, y, explanation)


class DecisionTreeOracle(Oracle):
    def __init__(self, problem, num_features=10):
        super().__init__(problem)

        from sklearn.tree import DecisionTreeClassifier
        # NOTE: the DT must be trained on the interpretable representation
        self.oracle = DecisionTreeClassifier(random_state=0) \
                          .fit(problem.Z, problem.y)

        perfs = prfs(problem.y, self.oracle.predict(problem.Z),
                     average='weighted')[:3]
        print('oracle performance =', perfs)
        print('oracle rules =\n' + \
              '\n'.join(self.get_rules(self.oracle.tree_)))

        self.num_features = num_features

    def get_rules(self, tree, booleanize=True):
        import sklearn

        def recurse(node, rules, prefix):
            l_child = tree.children_left[node]
            r_child = tree.children_right[node]
            if l_child == sklearn.tree._tree.TREE_LEAF:
                value = tree.value[node]
                rules.append(' ^ '.join(prefix) + ' -> ' + str(value))
            else:
                feature = tree.feature[node]
                feature_name = self.problem.feature_names[feature]
                threshold = tree.threshold[node]
                if not booleanize:
                    l_cond = '({} <= {})'.format(feature_name, threshold)
                    r_cond = '({} > {})'.format(feature_name, threshold)
                else:
                    assert threshold == 0.5
                    l_cond = '(not {})'.format(feature_name, threshold)
                    r_cond = '({})'.format(feature_name, threshold)
                recurse(l_child, rules, prefix + [l_cond])
                recurse(r_child, rules, prefix + [r_cond])

        rules = []
        recurse(0, rules, [])
        return rules

    def evaluate_explanation(self, example, y, pred_explanation):
        if pred_explanation is None:
            return -1, -1, -1, -1

        import sklearn

        z = self.problem.Z[example]
        tree = self.oracle.tree_

        current = 0
        l_child = tree.children_left[current]
        features_in_true_branch = set()
        while l_child != sklearn.tree._tree.TREE_LEAF:
            l_child = tree.children_left[current]
            r_child = tree.children_right[current]
            i = tree.feature[current]
            if z[i] != 0:
                features_in_true_branch.add(i)
            if len(features_in_true_branch) >= self.num_features:
                break
            if z[i] < tree.threshold[current]:
                current = l_child
            else:
                current = r_child

        true_explanation = [
                (self.problem.feature_names[i], np.sign(z[i]))
                for i in features_in_true_branch
            ]

        perfs = self.problem.get_explanation_perf(true_explanation,
                                                  pred_explanation.as_list())
        return perfs + (pred_explanation.score,)


class SparseLogRegOracle(Oracle):
    def __init__(self, problem, C=1, num_features=10, min_coef=1e-6):
        super().__init__(problem)

        from sklearn.linear_model import LogisticRegression
        # TODO crossvalidate C
        self.oracle = LogisticRegression(penalty='l1', C=C, random_state=0) \
                          .fit(problem.Z, problem.y)

        perfs = prfs(problem.y, self.oracle.predict(problem.Z),
                     average='weighted')[:3]
        print('oracle performance =', perfs)

        self.num_features = num_features
        self.min_coef = min_coef

    def evaluate_explanation(self, example, y, pred_explanation):
        if pred_explanation is None:
            return -1, -1, -1, -1

        abs_coef = np.abs(self.oracle.coef_[0])
        indices = abs_coef.argsort()[::-1][:self.num_features]

        true_explanation = [
                (self.problem.feature_names[i], self.oracle.coef_[0,i])
                for i in indices if abs_coef[i] >= self.min_coef
            ]

        perfs = self.problem.get_explanation_perf(true_explanation,
                                                  pred_explanation.as_list())
        return perfs + (pred_explanation.score,)


class LIMEOracle(Oracle):
    """NOTE: this evaluator requires a wrapped oracle."""

    def __init__(self, problem):
        super().__init__(problem)

    def evaluate_explanation(self, example, y, explanation):
        raise NotImplementedError()

        # XXX works around LIME usage of the default RNG
        np.random.seed(0)

        explanation = \
            self.problem.explain(self.oracle,
                                 self.problem.examples,
                                 example, y,
                                 num_samples=self.num_samples,
                                 num_features=self.num_features)


        perf = self.problem.get_explanation_perf(oracle_explanation,
                                                 explanation)
        return perf + (explanation.score,)


class TicTacToeOracle(Oracle):
    def __init__(self, problem):
        super().__init__(problem)

    @staticmethod
    def to_board(z):
        board = np.zeros((3, 3), dtype=str)
        for i, j in product(range(3), repeat=2):
            n = 9*i + j*3
            board[i,j] = {0: 'b', 1: 'x', 2: 'o'}[np.where(z[n:n+3])[0][0]]
        return board

    def evaluate_explanation(self, example, y, pred_explanation):
        if pred_explanation is None:
            return -1, -1, -1, -1

        board = np.array(self.problem._boards[example], dtype=str)

        WINNING_PIECES = (
            ['x', 'x', 'x'],
            ['x', 'x', ' '],
            ['x', ' ', 'x'],
            [' ', 'x', 'x'],
        )

        triplets = []
        for i in range(3):
            triplets.append([[i, 0], [i, 1], [i, 2]])
        for j in range(3):
            triplets.append([[0, j], [1, j], [2, j]])
        triplets.append([[0, 0], [1, 1], [2, 2]])
        triplets.append([[0, 2], [1, 1], [2, 0]])

        # XXX messy, much easier to do with an intersection
        relevant_features = set()
        for triplet in triplets:
            board_pieces = [board[i,j] for i, j in triplet]
            for pieces in WINNING_PIECES:
                if board_pieces == pieces:
                    for i, j in triplet:
                        s = board[i,j]
                        relevant_features.add('{i} {j} {s}'.format(**locals()))

        true_explanation = [
                (feature_name, 1)
                for feature_name in relevant_features
            ]

        perfs = self.problem.get_explanation_perf(true_explanation,
                                                  pred_explanation.as_list())
        return perfs + (pred_explanation.score,)
