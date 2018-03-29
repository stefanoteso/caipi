Caipi
=====

Caipirinha turns LIME into trust!  An implementation of the Caipi framework
for interactive explanatory learning.


Required Packages
-----------------

Caipi is written in Python 3.5.  Make sure that you have the following
packages:

- [numpy](https://www.numpy.org)
- [sklearn](https://scikit-learn.org)
- [lime](https://github.com/marcotcr/lime)
- [blessings](https://pypi.python.org/pypi/blessings)
- [nltk](http://www.nltk.org/) for the 20 newsgroups task
- [skimage](http://scikit-image.org/) for the image classification task


Usage
-----

You can run Caipi as follows:
```bash
    python3 caipi.py $problem $learner $example-selection-strategy
```
For the complete list of options, type:
```bash
    python3 caipi.py --help
```


Datasets
--------
The tic-tac-toe endgame dataset can be found at
[UCI](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)

The FER'13 emotion recognition dataset can be found on the
[Kaggle challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
page. Place the uncompressed data into the `data/` directory and run the
`fer13_preprocess.py` script.
