import numpy as np
from sklearn.pipeline import Pipeline
from skimage.color import gray2rgb, rgb2gray
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline

from lime.lime_image import LimeImageExplainer
from blessings import Terminal

from .problems import Problem
from .utils import PipeStep

from keras import applications

CNN_MODELS = {'vgg16': applications.VGG16,
              'vgg19': applications.VGG19,
              'resnet50': applications.ResNet50,
              'xception': applications.Xception,
              'inception': applications.InceptionV3}

_TERM = Terminal()


class _ImageProblem(Problem):
    # def __init__(self, *args, data, target, class_names, labels=None,
    #              noise=False, **kwargs):

    #     self.class_names = class_names
    #     self.labels = labels or list(range(class_names))
    #     indices = np.where(np.isin(target, self.labels))

    #     images, y = data[indices], target[indices]
    #     if noise:
    #         images = self.add_correlated_noise(images, y)

    #     self.examples = list(range(len(indices)))
    #     self.y = y
    #     self.X = np.stack([gray2rgb(image) for image in images], 0)

    def __init__(self, *args, y, X, X_lime, class_names, labels=None,
                 noise=False, kernel_size=1, ascii=True, wrap_img_preproc=False, **kwargs):

        self.class_names = class_names
        self.labels = y  # labels or list(range(class_names))
        # indices = np.where(np.isin(target, self.labels))

        # images, y = data[indices], target[indices]
        if noise:
            X = self.add_correlated_noise(X, y)

        self.examples = list(range(len(X)))
        self.y = y
        self.X = X
        self.X_lime = X_lime

        self.kernel_size = kernel_size
        self.ascii = ascii
        self.wrap_img_preproc = wrap_img_preproc

    def add_correlated_noise(self, images, y):
        """Adds a bunch of features correlated to the label."""
        noisy_images = []
        for image, label in zip(images, y):
            noise = np.zeros_like(image)
            height = range(2 * label, 2 * label + 2)
            noise[np.ix_(height, range(28))] = 255
            noisy_images.append(np.hstack((image, noise)))
        return np.array(noisy_images, dtype=np.uint8)

    def wrap_preproc_(self, model):
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

    def wrap_preproc(self, model):
        return model

    # def wrap_preproc(self, model):
    #     """Wraps a model into the preprocessing pipeline, if any."""
    #     assert not isinstance(model, Pipeline)
    #     # Converts from RGB images (i.e. self.X) to grayscale 1D vectors
    #     # TODO do this only once during initialization
    #     # TODO cache the result
    #     return Pipeline([
    #         # ('grayscale',
    #         #  PipeStep(lambda X: np.array([rgb2gray(x) for x in X]))),
    #         ('flatten',
    #          PipeStep(lambda X: np.array([x.ravel() for x in X]))),
    #         ('model', model)
    #     ])

    def explain(self, learner, known_examples, example, y,
                num_samples=500, num_features=10, **kwargs):
        from sklearn.linear_model import Ridge

        explainer = LimeImageExplainer(verbose=False)

        if self.wrap_img_preproc:
            learner = make_pipeline(self.wrap_preproc_exp(), learner)
            # print('LEARNER wrapped in preprocessing:', learner)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        explanation = \
            explainer.explain_instance(self.X_lime[example],
                                       classifier_fn=learner.predict_proba,
                                       model_regressor=local_model,
                                       top_labels=len(self.labels),
                                       num_samples=num_samples,
                                       hide_color=0,
                                       qs_kernel_size=self.kernel_size)

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
        return self.y[example]

    @staticmethod
    def asciiart(image, mask=None):
        asciiart = ''
        for i, row in enumerate(rgb2gray(image)):
            for j, value in enumerate(row):
                gray = 232 + int(round((1 - value) * 23))
                color, char = _TERM.blue, ' '
                if mask is not None and mask[i, j]:
                    char = '□'
                    color = [None, _TERM.red, _TERM.green][mask[i, j]]
                asciiart += (_TERM.on_color(gray) +
                             color +
                             _TERM.bold +
                             char +
                             _TERM.normal)
            asciiart += '\n'
        return asciiart

    def improve_explanation(self, example, y, explanation, num_features=10):
        """
        ASCII-art is the future.
        The future is past.
        """

        print('The model thinks that this picture is ' +
              "'" + _TERM.bold + _TERM.blue + self.class_names[y] + _TERM.normal + "'" +
              ' because of the '
              + _TERM.bold + _TERM.green + 'green' + _TERM.normal + ' pixels:\n')

        mask = explanation.masks[self.labels.index(y)]
        if self.ascii:
            image = self.X_lime[example]
            print(self.asciiart(rgb2gray(image), mask=mask))
        else:
            raise NotImplementedError('Yet to implement a non ascii visualizer')

    def get_explanation_perf(self, true_explanation, pred_explanation):
        def clamp(mask):
            mask[mask == 1] = 0
            mask[mask == 2] = 1
            return mask
        index = self.labels.index(pred_explanation.y)
        true_mask = clamp(true_explanation.masks[index].ravel())
        pred_mask = clamp(pred_explanation.masks[index].ravel())
        return recall_score(true_mask, pred_mask)


# (self, *args, y, X, X_lime, class_names, labels=None,
#                 noise = False, kernel_size = 1, ascii = True, **kwargs)
class MNISTProblem(_ImageProblem):
    def __init__(self, *args, **kwargs):
        from sklearn.datasets import fetch_mldata

        dataset = fetch_mldata('MNIST original')
        # super().__init__(*args,
        #                  data=dataset.data.reshape((-1, 28, 28)),
        #                  target=dataset.target.astype(np.uint8),
        #                  class_names=list(map(str, range(10))),
        #                  **kwargs)
        data = dataset.data.reshape((-1, 28, 28))

        super().__init__(*args, X=data, X_lime=data,
                         y=dataset.target.astype(np.uint8),
                         class_names=list(map(str, range(10))),
                         kernel_size=1, ascii=True,
                         **kwargs)


class FER13Problem(_ImageProblem):
    def __init__(self, *args, **kwargs):
        from .utils import load

        dataset = load('data/fer2013.pickle')

        # super().__init__(*args,
        #                  data=(255 - dataset['data']),
        #                  target=dataset['target'],
        #                  class_names=dataset['class_names'],
        #                  **kwargs)
        data = (255 - dataset['data'])
        super().__init__(*args,
                         X=data,
                         X_lime=data,
                         y=dataset['target'],
                         class_names=dataset['class_names'],
                         kernel_size=1, ascii=True,
                         **kwargs)


class PascalVOC2012Problem(_ImageProblem):
    def __init__(self, data_path='VOC2012/100/images-100x100.pklz',
                 feature_path='VOC2012/100/vgg16-100x100x3.pklz',
                 label_path='VOC2012/100/class-names.txt',
                 pretrained_model='vgg16',
                 aggregate='avg',
                 batch_size=64,
                 verbose=1,
                 * args, **kwargs):

        def load_class_names(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                return [l.split(',')[0].strip() for l in lines]

        from .utils import load_gzip
        #
        #
        class_names = load_class_names(label_path)
        print('Class names', class_names)

        feature_data, _y = load_gzip(feature_path)
        n_samples = feature_data.shape[0]

        image_data, y = load_gzip(data_path)
        assert len(image_data) == n_samples, len(image_data)

        # image_data = image_data.reshape((n_samples, -1))

        img_rows, img_cols, img_channels = image_data.shape[1:]

        self.aggregate = aggregate
        self.batch_size = batch_size
        self.verbose = verbose
        self.pretrained_model = pretrained_model
        #
        # loading deep cnn pretrained on imagenet
        self.imagenet_model = CNN_MODELS[self.pretrained_model](weights='imagenet',
                                                                include_top=False,
                                                                input_shape=(img_rows,
                                                                             img_cols,
                                                                             img_channels),
                                                                pooling=self.aggregate)

        self.imagenet_model.summary()

        super().__init__(*args,
                         # X=image_data,  # feature_data,
                         X=feature_data,
                         X_lime=image_data,
                         y=y,
                         class_names=class_names,
                         kernel_size=1, ascii=True,
                         wrap_img_preproc=True,
                         **kwargs)

    # def wrap_preproc_exp(self, model):
    def wrap_preproc_exp(self):
        """Wraps a model into the preprocessing pipeline, if any."""
        # assert not isinstance(model, Pipeline)
        #
        # extract features from CNN pretrained on imagenet
        # return Pipeline([
        #     ('imagenet-features',
        #      PipeStep(lambda X:self.imagenet_model.predict(X, verbose=self.verbose,
        #                                                    batch_size=self.batch_size))),
        #     ('model', model)
        # ])
        return PipeStep(lambda X: self.imagenet_model.predict(X, verbose=self.verbose,
                                                              batch_size=self.batch_size))
