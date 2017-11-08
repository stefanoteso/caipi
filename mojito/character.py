import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from skimage.color import gray2rgb, rgb2gray, label2rgb
from lime.lime_image import LimeImageExplainer

from .problems import Problem
from .utils import PipeStep


class CharacterProblem(Problem):
    """Character classification.

    Partially ripped from https://github.com/marcotcr/lime
    """
    def __init__(self, labels=None, rng=None):
        dataset = fetch_mldata('MNIST original')

        self.Y = dataset.target.astype(np.uint8)
        self.X = np.stack([
                gray2rgb(img) for img in dataset.data.reshape((-1, 28, 28))
            ], 0)
        self.examples = list(range(len(self.Y)))

    def explain(self, learner, train_examples, example,
                num_samples=5000, num_features=10):
        explainer = LimeImageExplainer(verbose=True)

        local_model = Ridge(alpha=1, fit_intercept=True)
        pipeline = Pipeline([
                ('', PipeStep(lambda X: [rgb2gray(x) for x in X])),
                ('', PipeStep(lambda X: [x.ravel() for x in X])),
                ('', learner.model_),
            ])
        explanation = explainer.explain_instance(self.X[example],
                                                 classifier_fn=pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_samples=num_samples,
                                                 hide_color=0,
                                                 qs_kernel_size=1)
        explanation.discrepancy = -1
        return explanation

    def improve(self, example, y):
        return self.Y[example]

    def improve_explanation(self, example, y, explanation):
        image, mask = explanation.get_image_and_mask(y,
                                                     num_features=10,
                                                     hide_rest=False,
                                                     min_weight=0.01)

        class_color = TextMod.BOLD + TextMod.GREEN if y else TextMod.RED
        class_name = class_color + str(y) + TextMod.END

        print('The model thinks that this instance:')
        print(self.X_lime[example])
        print('is {}, because of these values:'.format(class_name))
        for constraint, coeff in explanation.as_list():
            color = TextMod.RED if coeff < 0 else TextMod.GREEN
            coeff = TextMod.BOLD + color + '{:+3.1f}'.format(coeff) + TextMod.END
            print('  {:40s} : {}'.format(constraint, coeff))
