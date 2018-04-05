from sklearn.utils import check_random_state


class Problem:
    def __init__(self, n_samples, n_features, kernel_width, metric='euclidean',
                 rng=None):
        self.rng = check_random_state(rng)

        self.n_samples = n_samples
        self.n_features = n_features
        self.kernel_width = kernel_width
        self.metric = metric

    def explain(self, learner, known_examples, i, y_pred):
        """Computes the learner's explanation of a prediction."""
        raise NotImplementedError()

    def query_label(self, i):
        """Queries the oracle for a label."""
        raise NotImplementedError()

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl):
        """Queries the oracle for an improved explanation."""
        raise NotImplementedError()

    def save_expl(self, path, i, pred_y, expl):
        """Saves an explanation to file."""
        raise NotImplementedError()

    def eval(self, learner, known_examples, test_examples, eval_examples,
             t=None, basename=None):
        """Evaluates the learner."""
        raise NotImplementedError()
