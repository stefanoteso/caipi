import numpy as np
import time, re, blessings
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support as prfs
from lime.lime_text import LimeTextExplainer

from . import Problem
from . import load, densify, vstack, hstack


_TERM = blessings.Terminal()


class TextProblem(Problem):
    def explain(self, learner, known_examples, i, y_pred):
        explainer = LimeTextExplainer(class_names=self.class_names)

        pipeline = make_pipeline(self.vectorizer, learner)
        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        expl = explainer.explain_instance(self.processed_docs[i],
                                          pipeline.predict_proba,
                                          model_regressor=local_model,
                                          num_features=self.n_features,
                                          num_samples=self.n_samples)
        return expl

    def query_label(self, i):
        return self.y[i]

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl, X_test):
        if pred_expl is None:
            return X_corr, y_corr

        true_words = {word for word, _ in self.explanations[i]}
        if not len(true_words):
            # No explanation known for this example
            return X_corr, y_corr
        pred_words = {word for word, _ in pred_expl.as_list()}

        words_in_doc = set(self.processed_docs[i].split())

        corrected_docs = []
        for word in pred_words - true_words:
            corrected_docs.append(' '.join(words_in_doc - set([word])))

        if not len(corrected_docs):
            return X_corr, y_corr

        X_new_corr = self.vectorizer.transform(corrected_docs)
        y_new_corr = np.array([pred_y for _ in corrected_docs], dtype=np.int8)

        X_corr = vstack([X_corr, X_new_corr])
        y_corr = hstack([y_corr, y_new_corr])
        return X_corr, y_corr

    @staticmethod
    def _highlight_words(text, expl):
        for word, coeff in expl:
            color = _TERM.green if coeff >= 0 else _TERM.red
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
            fp.write(self._highlight_words(self.docs[i], expl))
            fp.write('\n' + 80 * '-' + '\n')
            fp.write('explanation:\n')
            for word, coeff in expl:
                fp.write('{:32s} : {:3.1f}\n'.format(word, coeff))

    def _eval_expl(self, learner, known_examples, eval_examples,
                   t=None, basename=None):
        if eval_examples is None:
            return -1, -1, -1

        perfs = []
        for i in eval_examples:
            true_y = self.y[i]
            true_expl = self.explanations[i]

            pred_y = learner.predict(densify(self.X[i]))[0]
            pred_expl = self.explain(learner, known_examples, i, pred_y)
            pred_expl = [(feat, int(np.sign(coeff)))
                         for feat, coeff in pred_expl.as_list()]

            matches = set(map(tuple, true_expl)).intersection(set(pred_expl))
            pr = len(matches) / len(pred_expl) if len(pred_expl) else 0.0
            rc = len(matches) / len(true_expl) if len(true_expl) else 0.0
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
        return tuple(pred_perfs) + tuple(expl_perfs)


class NewsgroupsProblem(TextProblem):
    def __init__(self, *args, classes=None, min_words=10, **kwargs):
        super().__init__(*args, **kwargs)

        path = join('data', '20newsgroups_{}_{}.pickle'.format(
                        '+'.join(sorted(classes)), min_words))
        try:
            dataset = load(path)
        except:
            raise RuntimeError('Run the data preparation script first!')

        self.class_names = classes
        self.y = dataset.target
        self.docs = dataset.data
        self.processed_docs = dataset.processed_data
        self.explanations = dataset.explanations

        self.vectorizer = TfidfVectorizer(lowercase=False).fit(self.processed_docs)
        self.X = self.vectorizer.transform(self.processed_docs)


class ReviewsProblem(TextProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        path = join('data', 'review_polarity_rationales.pickle')
        try:
            dataset = load(path)
        except:
            raise RuntimeError('Run the data preparation script first!')

        self.class_names = ['neg', 'pos']
        self.y = dataset['y']
        self.docs = self.processed_docs = dataset['docs']
        self.explanations = dataset['explanations']

        self.vectorizer = TfidfVectorizer(lowercase=False) \
                              .fit(self.processed_docs)
        self.X = self.vectorizer.transform(self.processed_docs)
