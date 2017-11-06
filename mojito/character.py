import numpy as np
from sklearn.datasets import fetch_mldata
from skimage.color import gray2rgb, rgb2gray, label2rgb

from .problems import Problem

class CharacterProblem(Problem):
    """Character classification.

    Partially ripped from http://scikit-learn.org/stable/datasets/
    """
    def __init__(self, labels=None, rng=None):
        dataset = fetch_mldata('MNIST original')

        self.Y = dataset.target.astype(np.uint8)
        self.X = np.stack([
                gray2rgb(img) for img in dataset.data.reshape((-1, 28, 28))
            ], axis=0)
        self.examples = list(range(len(self.Y)))

    def explain(self, learner, train_examples, examples, num_samples=5000):
        raise NotImplementedError()

    def improve(self, example, y):
        return self.Y[example]

    def improve_explanation(self, example, y, explanation):
        print(expalanation.as_list())
        return explanation
