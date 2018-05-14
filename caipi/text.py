import numpy as np
import re, blessings
from time import time
from os.path import join
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.utils import check_random_state
from lime.lime_text import LimeTextExplainer
from gensim.models.keyedvectors import KeyedVectors

from . import Problem, load, densify, vstack, hstack


_TERM = blessings.Terminal()


class Normalizer:
    def fit(self, X, y=None):
        return self

    def transform(self, X, norms=None, return_norms=False, append_value=1):
        new_X, ret_norms = [], []
        try:
            iterator = enumerate(X.todense())
        except:
            iterator = enumerate(X)
        for i, x in iterator:
            x = np.array(x).ravel()
            norm = np.linalg.norm(x) if norms is None else norms[i]
            norm = max(1e-16, norm)
            ret_norms.append(norm)
            new_X.append(x.ravel() / norm)
        new_X = np.array(new_X)
        if append_value is not None:
            new_X = np.hstack([new_X,
                               append_value * np.ones((new_X.shape[0], 1))])
        new_X = csr_matrix(np.array(new_X))
        if return_norms:
            return new_X, ret_norms
        return new_X


class Word2VecVectorizer:
    def __init__(self, path):
        self.word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
        self.n_features = self.word2vec.wv['the'].shape[0]
        self.no_embedding = set()

    def _embed_document(self, text):
        word_vectors = []
        for word in text.split():
            word = word.lower()
            try:
                word_vectors.append(self.word2vec.wv[word])
            except KeyError:
                if word not in self.no_embedding:
                    print('Warning: could not embed "{}"'.format(word))
                    self.no_embedding.add(word)
        return (np.mean(word_vectors, axis=0) if len(word_vectors) else
                np.zeros(self.n_features))

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        return np.array([self._embed_document(doc) for doc in docs])


class TextProblem(Problem):
    def __init__(self, **kwargs):
        self.class_names = kwargs.pop('class_names')
        self.y = kwargs.pop('y')
        self.docs = kwargs.pop('docs')
        self.processed_docs = kwargs.pop('processed_docs')
        self.explanations = kwargs.pop('explanations')
        self.lime_repeats = kwargs.pop('lime_repeats', 1)
        n_examples = kwargs.pop('n_examples', None)
        self.corr_type = kwargs.pop('corr_type', 'replace-expl') or 'replace-expl'
        self.vect_type = kwargs.pop('vect_type', 'glove')
        super().__init__(**kwargs)

        if n_examples is not None:
            rng = check_random_state(kwargs.get('rng', None))
            perm = rng.permutation(len(self.y))[:n_examples]
            self.y = self.y[perm]
            self.docs = [self.docs[i] for i in perm]
            self.processed_docs = [self.processed_docs[i] for i in perm]
            self.explanations = [self.explanations[i] for i in perm]

        self.normalizer = Normalizer()
        if self.vect_type == 'binary':
            self.vectorizer = CountVectorizer(lowercase=False, binary=True) \
                                  .fit(self.processed_docs)
        elif self.vect_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(lowercase=False) \
                                  .fit(self.processed_docs)
        elif self.vect_type == 'glove':
            path = join('data', 'word2vec_glove.6B.300d.bin')
            self.vectorizer = Word2VecVectorizer(path)
        elif self.vect_type == 'google-news':
            path = join('data', 'GoogleNews-vectors-negative300.bin')
            self.vectorizer = Word2VecVectorizer(path)
        else:
            raise ValueError('unknown vect_type "{}"'.format(self.vect_type))

        self.X = self.vectorizer.transform(self.processed_docs)
        self.X, self.norms = self.normalizer.transform(self.X,
                                                       return_norms=True,
                                                       append_value=1)

        self.explainable = {i for i in range(len(self.y))
                            if self.explanations[i] is not None and len(self.explanations[i])}

    def _masks_to_expl(self, i):
        """Turns a list of word masks (which might highlight repeated words)
        into a set of words."""
        words = self.processed_docs[i].split()
        true_masks = self.explanations[i]
        true_mask = np.sum(true_masks, axis=0)
        selected_words = {words[i] for i in np.where(true_mask)[0]}
        return [(word, self.y[i]) for word in selected_words]

    def explain(self, learner, known_examples, i, y_pred):
        explainer = LimeTextExplainer(class_names=self.class_names)

        # XXX hack
        if self.explanations[i] is None:
            n_features = 1
        else:
            true_expl = self._masks_to_expl(i)
            n_features = max(len(true_expl), 1) # XXX FIXME XXX

        pipeline = make_pipeline(self.vectorizer, self.normalizer, learner)

        counts = defaultdict(int)
        for r in range(self.lime_repeats):

            t = time()
            local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
            expl = explainer.explain_instance(self.processed_docs[i],
                                              pipeline.predict_proba,
                                              model_regressor=local_model,
                                              num_features=n_features,
                                              num_samples=self.n_samples)
            print('  LIME {}/{} took {:3.2f}s'.format(r + 1, self.lime_repeats,
                                                 time() - t))

            for word, coeff in expl.as_list():
                counts[(word, int(np.sign(coeff)))] += 1

        sorted_counts = sorted(counts.items(), key=lambda _: _[-1])
        sorted_counts = list(sorted_counts)[-n_features:]
        return [ws for ws, _ in sorted_counts]

    def query_label(self, i):
        return self.y[i]

    def query_corrections(self, i, pred_y, pred_expl, X_test):
        if pred_expl is None:
            return set()
        if pred_y != self.y[i]:
            return set()
        if i not in self.explainable:
            # XXX makes perf drop for some reason
            return set()

        all_words = set(self.processed_docs[i].split())
        true_words = {word for word, _ in self._masks_to_expl(i)}
        pred_words = {word for word, _ in pred_expl}
        fp_words = pred_words - true_words

        print('original =', ' '.join(all_words))
        if self.corr_type == 'replace-expl':
            correction = ' '.join(true_words)
            print('correction =', correction)
            print()
            self.X[i] = self.normalizer.transform(
                            self.vectorizer.transform([correction]))[0]
            self.processed_docs[i] = correction
            extra_examples = set()

        elif self.corr_type == 'replace-no-fp':
            correction = ' '.join(all_words - fp_words)
            print('correction =', correction)
            print()
            self.X[i] = self.normalizer.transform(
                            self.vectorizer.transform([correction]))[0]
            self.processed_docs[i] = correction
            extra_examples = set()

        elif self.corr_type == 'add-contrast':
            sorted_words = np.array(self.processed_docs[i].split())

            # TODO corrections should have x_final = 0

            corrections = []
            for mask in self.explanations[i]:
                masked_indices = np.where(mask)[0]
                rationale = ' '.join(sorted_words[masked_indices])
                corrections.append(rationale)

            X_corrections = self.vectorizer.transform(corrections)
            X_corrections = self.normalizer.transform(X_corrections,
                                                      norms=[self.norms[i] for _ in corrections],
                                                      append_value=0)

            extra_examples = set(range(self.X.shape[0],
                                       self.X.shape[0] + len(corrections)))

        else:
            raise ValueError('unknown correction type "{}"'.format(
                                 self.corr_type))

        if len(extra_examples):
            y_corrections = np.array([pred_y] * len(corrections))
            self.X = vstack([self.X, X_corrections])
            self.y = hstack([self.y, y_corrections])

        return extra_examples

    def _eval_expl(self, learner, known_examples, eval_examples,
                   t=None, basename=None):
        if eval_examples is None:
            return -1, -1, -1

        perfs = []
        for i in set(eval_examples) & self.explainable:
            true_y = self.y[i]
            true_expl = self._masks_to_expl(i)

            pred_y = learner.predict(densify(self.X[i]))[0]
            pred_expl = self.explain(learner, known_examples, i, pred_y)

            # NOTE here we don't care if the coefficients are wrong, since
            # those depend on whether the prediction is wrong

            true_words = {(word, np.sign(coeff)) for word, coeff in true_expl}
            pred_words = {(word, np.sign(coeff) if true_y == pred_y else -np.sign(coeff))
                          for word, coeff in pred_expl}

            matches = true_words & pred_words
            pr = len(matches) / len(pred_words) if len(pred_words) else 0.0
            rc = len(matches) / len(true_words) if len(true_words) else 0.0
            f1 = 0.0 if pr + rc <= 0 else 2 * pr * rc / (pr + rc)
            perfs.append((pr, rc, f1))

            if basename is None:
                continue

            self.save_expl(basename + '_{}_{}.txt'.format(i, t),
                           i, pred_y, pred_expl)
            self.save_expl(basename + '_{}_true.txt'.format(i),
                           i, true_y, true_expl)

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
        return np.array(tuple(pred_perfs) + tuple(expl_perfs))

    @staticmethod
    def _highlight_words(text, expl):
        for word, sign in expl:
            color = _TERM.green if sign >= 0 else _TERM.red
            colored_word = color + word + _TERM.normal
            matches = list(re.compile(word).finditer(text))
            matches.reverse()
            for match in matches:
                start = match.start()
                text = text[:start] + colored_word + text[start+len(word):]
        return text

    def save_expl(self, path, i, pred_y, expl):
        with open(path, 'wt') as fp:
            fp.write('true y: ' + self.class_names[self.y[i]] + '\n')
            fp.write('pred y: ' + self.class_names[pred_y] + '\n')
            fp.write(80 * '-' + '\n')
            #fp.write(self._highlight_words(self.docs[i], expl))
            fp.write('\n' + 80 * '-' + '\n')
            fp.write('explanation:\n')
            for word, sign in sorted(expl):
                fp.write('{:32s} : {:3.1f}\n'.format(word, sign))


class ReviewsProblem(TextProblem):
    def __init__(self, **kwargs):

        path = join('data', 'review_polarity_rationales.pickle')
        try:
            dataset = load(path)
        except:
            raise RuntimeError('Run the data preparation script first!')

        super().__init__(class_names=['neg', 'pos'],
                         y=dataset['y'],
                         docs=dataset['docs'],
                         processed_docs=dataset['docs'],
                         explanations=dataset['explanations'],
                         **kwargs)


class NewsgroupsProblem(TextProblem):
    def __init__(self, **kwargs):

        path = join('data', 'newsgroups.pickle')
        try:
            dataset = load(path)
        except:
            raise RuntimeError('Run the data preparation script first!')

        super().__init__(class_names=['neg', 'pos'],
                         y=dataset['y'],
                         docs=dataset['docs'],
                         processed_docs=dataset['docs'],
                         explanations=dataset['explanations'],
                         **kwargs)
