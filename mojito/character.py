import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from skimage.color import gray2rgb, rgb2gray
from sklearn.linear_model import Ridge
from sklearn.metrics import recall_score
from lime.lime_image import LimeImageExplainer
import matplotlib.pyplot as plt
from blessings import Terminal

from .problems import Problem
from .utils import PipeStep


_TERM = Terminal()


class CharacterProblem(Problem):
    """Character classification.

    Partially ripped from https://github.com/marcotcr/lime
    """
    def __init__(self, *args, labels=None, noise=True, rng=None, **kwargs):
        super().__init__(*args, **kwargs)

        dataset = fetch_mldata('MNIST original')

        self.labels = labels or tuple(range(10))

        y = dataset.target.astype(np.uint8)
        indices = np.where(np.isin(y, self.labels))

        y = y[indices]
        images = dataset.data[indices].reshape((-1, 28, 28))
        if noise:
            images = self.add_noise(images, y)

        self.Y = y
        self.X = np.stack([gray2rgb(image) for image in images], 0)
        self.examples = list(range(len(self.Y)))

    def add_noise(self, images, y):
        """Adds a diagonal feature correlated to the label."""
        noisy_images = []
        for image, label in zip(images, y):
            noise = np.zeros_like(image)
            height = range(2*label, 2*label + 2)
            noise[np.ix_(height, range(28))] = 255
            noisy_images.append(np.hstack((image, noise)))
        return np.array(noisy_images, dtype=np.uint8)

    def wrap_preproc(self, model):
        """Wraps a model into the preprocessing pipeline, if any."""
        assert not isinstance(model, Pipeline)
        # Converts from RGB images (i.e. self.X) to grayscale 1D vectors
        # TODO do this only once during initialization
        # TODO cache the result
        return Pipeline([
                ('grayscale',
                    PipeStep(lambda X: np.array([rgb2gray(x) for x in X]))),
                ('flatten',
                    PipeStep(lambda X: np.array([x.ravel() for x in X]))),
                ('model', model)
            ])

    def explain(self, learner, train_examples, example, y,
                num_samples=5000, num_features=10):
        explainer = LimeImageExplainer(verbose=False)
        explanation = \
            explainer.explain_instance(self.X[example],
                                       classifier_fn=learner.predict_proba,
                                       top_labels=len(self.labels),
                                       num_samples=num_samples,
                                       hide_color=0,
                                       qs_kernel_size=1)

        # Explain every label
        masks = []
        for i in range(len(self.labels)):
            _, mask = explanation.get_image_and_mask(i,
                                                     positive_only=False,
                                                     num_features=num_features,
                                                     min_weight=0.01,
                                                     hide_rest=False)
            masks.append(mask)
        explanation.masks = masks
        explanation.y = y

        return explanation

    def improve(self, example, y):
        return self.Y[example]

    @staticmethod
    def asciiart(image, mask=None):
        asciiart = ''
        for i, row in enumerate(rgb2gray(image)):
            for j, value in enumerate(row):
                gray = 232 + int(round((1 - value) * 23))
                color, char = _TERM.blue, ' '
                if mask is not None and mask[i,j]:
                    char = '□'
                    color = [None, _TERM.green, _TERM.red][mask[i,j]]
                asciiart += (_TERM.on_color(gray) +
                             color +
                             _TERM.bold +
                             char +
                             _TERM.normal)
            asciiart += '\n'
        return asciiart

    def improve_explanation(self, example, y, explanation, num_features=10):
        """ASCII-art is the future."""
        print('The model thinks that this picture is a ' +
               "'" + _TERM.bold + _TERM.blue + str(y) + _TERM.normal + "'" +
               ' because of the '
               + _TERM.bold + _TERM.red + 'red' + _TERM.normal + ' pixels:\n')
        image = self.X[example]
        mask = explanation.masks[self.labels.index(y)]
        print(self.asciiart(rgb2gray(image), mask=mask))

    def get_explanation_perf(self, true_explanation, pred_explanation):
        def clamp(mask):
            mask[mask == 1] = 0
            mask[mask == 2] = 1
            return mask
        index = self.labels.index(pred_explanation.y)
        true_mask = clamp(true_explanation.masks[index].ravel())
        pred_mask = clamp(pred_explanation.masks[index].ravel())
        return recall_score(true_mask, pred_mask)
