import pickle


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
