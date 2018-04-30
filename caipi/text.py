import numpy as np
import re, blessings
from time import time
from os.path import join
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support as prfs
from lime.lime_text import LimeTextExplainer

from . import Problem, load, densify, vstack, hstack


_TERM = blessings.Terminal()


class TextProblem(Problem):
    def __init__(self, **kwargs):
        self.class_names = kwargs.pop('class_names')
        self.y = kwargs.pop('y')
        self.docs = kwargs.pop('docs')
        self.processed_docs = kwargs.pop('processed_docs')
        self.explanations = kwargs.pop('explanations')
        self.lime_repeats = kwargs.pop('lime_repeats', 1)
        _ = kwargs.pop('n_examples', None)
        self.correction_method = kwargs.pop('correction_method', 'singleton')
        super().__init__(**kwargs)

        self.vectorizer = TfidfVectorizer(lowercase=False) \
                              .fit(self.processed_docs)
        self.X = self.vectorizer.transform(self.processed_docs)

        self.explainable = {i for i in range(len(self.y))
                            if len(self.explanations[i])}

    def explain(self, learner, known_examples, i, y_pred):
        explainer = LimeTextExplainer(class_names=self.class_names)

        pipeline = make_pipeline(self.vectorizer, learner)

        counts = defaultdict(int)
        for r in range(self.lime_repeats):

            t = time()
            local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
            expl = explainer.explain_instance(self.processed_docs[i],
                                              pipeline.predict_proba,
                                              model_regressor=local_model,
                                              num_features=self.n_features,
                                              num_samples=self.n_samples)
            print('  LIME {}/{} took {}s'.format(r + 1, self.lime_repeats,
                                                 time() - t))

            for word, coeff in expl.as_list():
                counts[(word, int(np.sign(coeff)))] += 1

        sorted_counts = sorted(counts.items(), key=lambda _: _[-1])
        sorted_counts = list(sorted_counts)[-self.n_features:]
        return [ws for ws, _ in sorted_counts]

    def query_label(self, i):
        return self.y[i]

    def query_corrections(self, X_corr, y_corr, i, pred_y, pred_expl, X_test):
        if pred_expl is None or not i in self.explainable:
            return X_corr, y_corr

        true_words = {word for word, _ in self.explanations[i]}
        pred_words = {word for word, _ in pred_expl}
        fp_words = pred_words - true_words

        words_in_doc = set(self.processed_docs[i].split())

        if self.correction_method == 'singleton':
            corrected_docs = [
                ' '.join(words_in_doc - fp_words)
            ]
        elif self.correction_method == 'words':
            corrected_docs = []
            for fp_word in fp_words:
                corrected_docs.append(' '.join(words_in_doc - set([fp_word])))
        elif self.correction_method == 'subsets':
            fp_words = np.array(list(sorted(fp_words)), dtype=str)

            corrected_docs = []
            for mask in self.rng.randint(0, 2, size=(10, len(fp_words))):
                print(mask)
                print(fp_words[mask])
                corrected_docs.append(' '.join(words_in_doc - set(fp_words[mask])))

        if not len(corrected_docs):
            return X_corr, y_corr

        X_new_corr = self.vectorizer.transform(corrected_docs)
        y_new_corr = np.array([pred_y for _ in corrected_docs], dtype=np.int8)

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
            true_expl = self.explanations[i]

            pred_y = learner.predict(densify(self.X[i]))[0]
            pred_expl = self.explain(learner, known_examples, i, pred_y)

            # NOTE here we don't care if the coefficients are wrong, since
            # those depend on whether the prediction is wrong

            true_words = {(word, np.sign(coeff)) for word, coeff in self.explanations[i]}
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
        return tuple(pred_perfs) + tuple(expl_perfs)

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
            fp.write(self._highlight_words(self.docs[i], expl))
            fp.write('\n' + 80 * '-' + '\n')
            fp.write('explanation:\n')
            for word, sign in expl:
                fp.write('{:32s} : {:3.1f}\n'.format(word, sign))


class NewsgroupsProblem(TextProblem):
    def __init__(self, *args, **kwargs):
        classes = kwargs.pop('classes', None)
        min_words = kwargs.pop('min_words', 10)

        path = join('data', '20newsgroups_{}_{}.pickle'.format(
                        '+'.join(sorted(classes)), min_words))
        try:
            dataset = load(path)
        except:
            raise RuntimeError('Run the data preparation script first!')

        super().__init__(class_names=classes,
                         y=dataset.target,
                         docs=dataset.data,
                         processed_docs=dataset.processed_data,
                         explanations=dataset.explanations,
                         **kwargs)


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
