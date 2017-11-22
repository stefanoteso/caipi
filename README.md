Mojito
======

An implementation of Coactive Learning with LIME on top.

Required Packages
-----------------

- [numpy](https://www.numpy.org)
- [sklearn](https://scikit-learn.org)
- [lime](https://github.com/marcotcr/lime)
- [blessings](https://pypi.python.org/pypi/blessings)
- [skimage](http://scikit-image.org/) for the image classification task
- [nltk](http://www.nltk.org/) for the 20 newsgroups task

Usage
-----
Run it like this:
```bash
    python3 run.py $problem $learner $example-selection-strategy
```
For the complete list of options, type:
```bash
    python3 run.py --help
```

Datasets
--------
For the FER'13 emotion recognition dataset, make sure to first download the
dataset from Kaggle into the `data/` directory and run the
`fer13_preprocess.py` script.
