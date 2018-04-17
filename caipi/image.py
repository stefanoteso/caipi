import numpy as np
import blessings
import matplotlib.pyplot as plt
from time import time
from itertools import product
from sklearn.datasets import fetch_mldata
from skimage.color import gray2rgb, rgb2gray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.utils import check_random_state
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm

from . import Problem, PipeStep, densify, setprfs, vstack, hstack


_TERM = blessings.Terminal()


class ImageProblem(Problem):
    def __init__(self, **kwargs):
        self.class_names = kwargs.pop('class_names')
        self.y = kwargs.pop('y')
        self.images = self._add_confounders(kwargs.pop('images'))
        self.lime_repeats = kwargs.pop('lime_repeats', 1)
        self.X = np.stack([gray2rgb(image) for image in self.images], 0)

        self.explainable = set(range(len(self.y)))

        super().__init__(**kwargs)

    def _add_confounders(self, images):
        noisy_images = []
        for image, label in zip(images, self.y):
            confounder = self._y_to_confounder(image, label)
            noisy_images.append(np.maximum(image, confounder))
        return np.array(noisy_images, dtype=np.uint8)

    def _y_to_confounder(self, image, label):
        dy = image.shape[-1] // len(self.class_names)
        xs, ys = range(label * dy, label * dy + 2), range(dy)
        mask = np.zeros_like(image)
        mask[np.ix_(xs, ys)] = 128
        return mask

    def _x_to_asciiart(self, x, mask=None, segments=None):
        text = ''
        for i, row in enumerate(rgb2gray(x)):
            for j, value in enumerate(row):
                gray = 232 + int(round((1 - value) * 23))
                color, char = _TERM.blue, ' '
                if mask is not None and mask[i,j] != 0:
                    char = '□'
                    color = {1: _TERM.red, 2: _TERM.green}[mask[i,j]]
                if segments is not None:
                    char = '□'
                    color = _TERM.color(segments[i, j] & 15)
                text += (_TERM.on_color(gray) +
                         color +
                         _TERM.bold +
                         char +
                         _TERM.normal)
            text += '\n'
        return text

    def preproc(self, images):
        return np.array([rgb2gray(image).ravel() for image in images])

    def explain(self, learner, known_examples, i, pred_y):
        explainer = LimeImageExplainer(verbose=False)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        # NOTE we *oversegment* the image on purpose!
        segmenter = SegmentationAlgorithm('quickshift',
                                          kernel_size=1,
                                          max_dist=10,
                                          ratio=0.2,
                                          sigma=0,
                                          random_seed=0)
        expl = explainer.explain_instance(self.X[i],
                                          classifier_fn=learner.predict_proba,
                                          segmentation_fn=segmenter,
                                          model_regressor=local_model,
                                          top_labels=len(self.class_names),
                                          num_samples=self.n_samples,
                                          hide_color=0)

        masks = []
        for target_y in range(len(self.class_names)):
            _, mask = expl.get_image_and_mask(target_y,
                                              positive_only=False,
                                              num_features=self.n_features,
                                              min_weight=0.01,
                                              hide_rest=False)
            masks.append(mask)

        return masks

    def query_label(self, i):
        return self.y[i]

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl, X_test):
        # NOTE we provide corrections regardless of the predicted label
        if pred_expl is None or not i in self.explainable:
            return X_corr, y_corr

        image = self.images[i]
        conf_mask = self._y_to_confounder(image, self.y[i])
        conf_mask[conf_mask == 128] = 2

        conf_coords = {(r, c)
                       for r in range(image.shape[0])
                       for c in range(image.shape[1])
                       if conf_mask[r,c] != 0}
        pred_coords = {(r, c)
                       for r in range(image.shape[0])
                       for c in range(image.shape[1])
                       if pred_expl[pred_y][r, c] != 0}
        fp_coords = conf_coords & pred_coords

        X_new_corr = []
        for value in [0, 255]:
            corr_image = np.array(image, copy=True)
            for r, c in fp_coords:
                corr_image[r, c] = value
            X_new_corr.append(gray2rgb(corr_image))

        X_new_corr = np.array(X_new_corr)
        y_new_corr = np.array([self.y[i]] * len(X_new_corr), dtype=np.int8)

        X_corr = vstack([X_corr, X_new_corr])
        y_corr = hstack([y_corr, y_new_corr])
        return X_corr, y_corr

    def _eval_expl(self, learner, known_examples, eval_examples,
                   t=None, basename=None):
        if eval_examples is None:
            return -1, -1, -1

        perfs = []
        for i in set(eval_examples) & self.explainable:
            true_y = self.y[i]
            pred_y = learner.predict(densify(self.X[i]))[0]

            image = self.images[i]
            conf_mask = self._y_to_confounder(image, true_y)
            conf_mask[conf_mask == 128] = 2

            pred_masks = self.explain(learner, known_examples, i, pred_y)

            # Compute pr/rc/f1 between confounders and positive pixels, we
            # want confounders to be used less and less with iterations
            conf_coords = {(r, c)
                           for r in range(image.shape[0])
                           for c in range(image.shape[1])
                           if conf_mask[r, c] != 0}
            pred_coords = {(r, c)
                           for r in range(image.shape[0])
                           for c in range(image.shape[1])
                           if pred_masks[pred_y][r, c] != 0}

            perfs.append(setprfs(conf_coords, pred_coords))

            if basename is None:
                continue

            self.save_expl(basename + '_{}_{}.png'.format(i, t),
                           i, pred_y, pred_masks[pred_y])
            self.save_expl(basename + '_{}_true.png'.format(i),
                           i, true_y, conf_mask)

        return np.mean(perfs, axis=0)

    def eval(self, learner, known_examples, test_examples, eval_examples,
             t=None, basename=None):
        pred_perfs = prfs(self.y[test_examples],
                          learner.predict(self.X[test_examples]),
                          average='weighted')[:3]
        expl_perfs = self._eval_expl(learner,
                                     known_examples,
                                     eval_examples,
                                     t=t, basename=basename)
        return tuple(pred_perfs) + tuple(expl_perfs)

    def save_expl(self, path, i, y, mask):
        from skimage.color import rgb2hsv, hsv2rgb

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')
        ax.text(0.5, 1.05,
                'true = {} | this = {}'.format(self.y[i], y),
                horizontalalignment='center',
                transform=ax.transAxes)

        overlay = np.zeros((mask.shape[0], mask.shape[1], 3))
        for r, c in product(range(mask.shape[0]), range(mask.shape[1])):
            if mask[r, c] == 1:
                overlay[r, c] = [1, 0, 0]
            if mask[r, c] == 2:
                overlay[r, c] = [0, 1, 0]
        overlay = rgb2hsv(overlay)

        masked_image = rgb2hsv(self.X[i])
        masked_image[..., 0] = overlay[..., 0] # hue
        masked_image[..., 1] = overlay[..., 1] * 0.6 # saturation
        masked_image = hsv2rgb(masked_image)

        ax.imshow(masked_image)

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)


class MNISTProblem(ImageProblem):
    def __init__(self, n_examples=100, **kwargs):
        mnist = fetch_mldata('MNIST original')

        images = mnist.data.reshape((-1, 28, 28))
        y = mnist.target.astype(np.uint8)

        if n_examples is not None:
            rng = check_random_state(kwargs.get('rng', None))
            perm = rng.permutation(len(y))[:n_examples]
            images, y = images[perm], y[perm]

        super().__init__(images=images,
                         y=y,
                         class_names=list(map(str, range(10))),
                         **kwargs)
