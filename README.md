Caipi
=====

An implementation of the CAIPI framework for interactive explanatory learning.


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

You can run CAIPI as follows:
```bash
    python3 caipi.py $problem $learner $example-selection-strategy
```
For the complete list of options, type:
```bash
    python3 caipi.py --help
```
