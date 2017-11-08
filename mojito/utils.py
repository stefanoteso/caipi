import pickle


class TextMod:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)


def densify(x):
    try:
        return x.toarray()
    except AttributeError:
        return x


class PipeStep:
    def __init__(self, func):
        self.func = func
    def fit(self, *args, **kwargs):
        return self
    def transform(self, X):
        return self.func(X)
