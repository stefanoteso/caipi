import numpy as np
import blessings
import matplotlib.pyplot as plt
import gzip
from os.path import join
from matplotlib.cm import get_cmap
from time import time
from itertools import product
from sklearn.datasets import fetch_mldata
from skimage.color import gray2rgb, rgb2gray, hsv2rgb, rgb2hsv
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.utils import check_random_state
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm

from . import Problem, PipeStep, densify, vstack, hstack


_TERM = blessings.Terminal()


class ImageProblem(Problem):
    def __init__(self, **kwargs):
        labels = kwargs.pop('labels')
        images = kwargs.pop('images')
        self.class_names = kwargs.pop('class_names')
        n_examples = kwargs.pop('n_examples', None)
        self.lime_repeats = kwargs.pop('lime_repeats', 1)

        if n_examples is not None:
            rng = check_random_state(kwargs.get('rng', None))
            perm = rng.permutation(len(labels))[:n_examples]
            images, labels = images[perm], labels[perm]

        self.y = labels
        self.images = self._add_confounders(images)
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
        dd = image.shape[-1] // len(self.class_names)
        ys, xs = range(label * dd, (label + 1) * dd), range(dd)
        mask = np.zeros_like(image)
        mask[np.ix_(ys, xs)] = 255
        return mask

    def preproc(self, images):
        return np.array([rgb2gray(image).ravel() for image in images])

    def explain(self, learner, known_examples, i, pred_y, return_segments=False):
        explainer = LimeImageExplainer(verbose=False)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        # NOTE we *oversegment* the image on purpose!
        segmenter = SegmentationAlgorithm('quickshift',
                                          kernel_size=1,
                                          max_dist=4,
                                          ratio=0.1,
                                          sigma=0,
                                          random_seed=0)
        expl = explainer.explain_instance(self.X[i],
                                          top_labels=len(self.class_names),
                                          classifier_fn=learner.predict_proba,
                                          segmentation_fn=segmenter,
                                          model_regressor=local_model,
                                          num_samples=self.n_samples,
                                          num_features=self.n_features,
                                          batch_size=1,
                                          hide_color=False)
        #print(expl.top_labels)
        _, mask = expl.get_image_and_mask(pred_y,
                                          positive_only=False,
                                          num_features=self.n_features,
                                          min_weight=0.01,
                                          hide_rest=False)
        if return_segments:
            return mask, expl.segments
        return mask

    def query_label(self, i):
        return self.y[i]

    @staticmethod
    def _extract_coords(image, mask):
        return {(r, c)
                for r in range(image.shape[0])
                for c in range(image.shape[1])
                if mask[r, c] != 0}

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_mask, X_test):
        true_y = self.y[i]
        if pred_mask is None or \
           pred_y != true_y or \
           not i in self.explainable:
            return X_corr, y_corr

        image = self.images[i]
        conf_mask = self._y_to_confounder(image, self.y[i])
        conf_mask[conf_mask == 255] = 2

        conf_coords = self._extract_coords(image, conf_mask)
        pred_coords = self._extract_coords(image, pred_mask)
        fp_coords = conf_coords & pred_coords

        X_new_corr = []
        for value in [-10, 0, 11]:
            corr_image = np.array(image, copy=True)
            for r, c in fp_coords:
                print('correcting pixel {},{} for label {}'.format(
                          r, c, true_y))
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
            return -1,

        perfs = []
        for i in set(eval_examples) & self.explainable:
            true_y = self.y[i]
            pred_y = learner.predict(densify(self.X[i]))[0]

            image = self.images[i]
            conf_mask = self._y_to_confounder(image, true_y)
            conf_mask[conf_mask == 255] = 2

            pred_mask, segments = \
                self.explain(learner, known_examples, i, pred_y,
                             return_segments=True)

            # Compute confounder recall
            conf_coords = self._extract_coords(image, conf_mask)
            pred_coords = self._extract_coords(image, pred_mask)
            perfs.append(len(conf_coords & pred_coords) / len(conf_coords))

            if basename is None:
                continue

            self.save_expl(basename + '_{}_true.png'.format(i),
                           i, true_y, mask=conf_mask)
            self.save_expl(basename + '_{}_{}_expl.png'.format(i, t),
                           i, pred_y, mask=pred_mask)

        return np.mean(perfs, axis=0),

    def eval(self, learner, known_examples, test_examples, eval_examples,
             t=None, basename=None):
        pred_perfs = learner.score(self.X[test_examples],
                                   self.y[test_examples]),
        expl_perfs = self._eval_expl(learner,
                                     known_examples,
                                     eval_examples,
                                     t=t, basename=basename)
        return tuple(pred_perfs) + tuple(expl_perfs)

    def save_expl(self, path, i, y, mask=None, segments=None):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')
        ax.text(0.5, 1.05,
                'true = {} | this = {}'.format(self.y[i], y),
                horizontalalignment='center',
                transform=ax.transAxes)

        cmap = get_cmap('tab20')

        r, c = self.images[i].shape
        if mask is not None:
            image = np.zeros((r, c, 3))
            for r, c in product(range(r), range(c)):
                image[r, c] = cmap((mask[r, c] & 3) / 3)[:3]
        elif segments is not None:
            image = np.zeros((r, c, 3))
            for r, c in product(range(r), range(c)):
                image[r, c] = cmap((segments[r, c] & 15) / 15)[:3]
        else:
            image = self.X[i]
        ax.imshow(image)

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)


def _load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as fp:
        labels = np.frombuffer(fp.read(), dtype=np.uint8, offset=8)

    images_path = join(path, '{}-images-idx3-ubyte.gz'.format(kind))
    with gzip.open(images_path, 'rb') as fp:
        images = np.frombuffer(fp.read(), dtype=np.uint8, offset=16)

    return images.reshape(len(labels), 28, 28), labels


class MNISTProblem(ImageProblem):
    def __init__(self, n_examples=None, **kwargs):
        path = join('data', 'mnist')
        tr_images, tr_labels = _load_mnist(path, kind='train')
        ts_images, ts_labels = _load_mnist(path, kind='t10k')
        images = np.vstack((tr_images, ts_images))
        labels = np.hstack((tr_labels, ts_labels))

        CLASS_NAMES = list(map(str, range(10)))

        super().__init__(images=images,
                         labels=labels,
                         class_names=CLASS_NAMES,
                         n_examples=n_examples,
                         **kwargs)


class FashionProblem(ImageProblem):
    def __init__(self, n_examples=None, **kwargs):
        path = join('data', 'fashion')
        tr_images, tr_labels = _load_mnist(path, kind='train')
        ts_images, ts_labels = _load_mnist(path, kind='t10k')
        images = np.vstack((tr_images, ts_images))
        labels = np.hstack((tr_labels, ts_labels))

        CLASS_NAMES = [
            'T-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
            'shirt', 'sneaker', 'bag', 'ankle_boots'
        ]

        super().__init__(images=images,
                         labels=labels,
                         class_names=CLASS_NAMES,
                         n_examples=n_examples,
                         **kwargs)
