import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from blessings import Terminal

from . import Problem, load, dump


_TERM = Terminal()


class NewsgroupsProblem(Problem):
    """Document classification.

    Partially ripped from https://github.com/marcotcr/lime
    """
    def __init__(self, *args, labels=None, min_words=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_words = min_words

        # TODO use standard 20newsgroups processing, ask Antonio

        from os.path import join

        path = join('cache', '20newsgroups.pickle')
        try:
            print('loading 20newsgroups...')
            dataset, self.documents = load(path)
        except:
            print('failed, preprocessing 20newsgroups...')
            # NOTE let's keep the quotes, they are pretty informative --
            # although maybe they leak test data in the training set?
            dataset = fetch_20newsgroups(subset='all',
                                         remove=('headers', 'footers'),
                                         random_state=0)
            self.documents = self.preprocess(dataset.data)

            print('caching preprocessed dataset...')
            dump(path, (dataset, self.documents))

        self.class_names = dataset.target_names
        if labels is None:
            self.labels = list(range(len(self.class_names)))
        else:
            self.labels = [self.class_names.index(label) for label in labels]
        indices = list(np.where(np.isin(dataset.target, self.labels))[0])

        self.examples = list(range(len(indices)))
        self.Y = dataset.target[indices]
        documents = [self.documents[i] for i in indices]
        self.vectorizer = TfidfVectorizer(lowercase=False).fit(documents)
        self.X = self.vectorizer.transform(documents)
        self.full_documents = [dataset.data[i] for i in indices]

    def wrap_preproc(self, model):
        return model

    @staticmethod
    def preprocess(data):
        """Reduces documents to lists of adjectives, nouns, and verbs."""
        import nltk

        VALID_TAGS = set([
            'FW',   # Foreign word
            'JJ',   # Adjective
            'JJR',  # Adjective, comparative
            'JJS',  # Adjective, superlative
            'NN',   # Noun, singular or mass
            'NNS',  # Noun, plural
            'NNP',  # Proper noun, singular
            'NNPS', # Proper noun, plural
            'UH',   # Interjection
            'VB',   # Verb, base form
            'VBD',  # Verb, past tense
            'VBG',  # Verb, gerund or present participle
            'VBN',  # Verb, past participle
            'VBP',  # Verb, non-3rd person singular present
            'VBZ',  # Verb, 3rd person singular present
        ])

        processed_data = []
        for i, text in enumerate(data):
            print('preprocessing document {} of {}'.format(i, len(data)))
            processed_text = ' '.join(token for token, tag
                                      in nltk.pos_tag(nltk.word_tokenize(text))
                                      if tag in VALID_TAGS)
            processed_data.append(processed_text)
        return processed_data

    def wrap_preproc(self, model):
        return model

    def explain(self, learner, train_examples, example, y,
                num_samples=5000, num_features=10):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline

        explainer = LimeTextExplainer(class_names=self.labels, verbose=True)

        local_model = Ridge(alpha=1, fit_intercept=True)
        pipeline = make_pipeline(self.vectorizer, learner)

        document = self.documents[example]
        if len(document.split()) < self.min_words:
            document = 'FOO BAR BAZ'
        explanation = explainer.explain_instance(document,
                                                 pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_features=num_features,
                                                 num_samples=num_samples)
        explanation.discrepancy = -1
        return explanation

    def improve(self, example, y):
        return self.Y[example]

    @staticmethod
    def highlight_words(text, explanation):
        import re

        for word, coeff in explanation.as_list():
            colored_word = _TERM.underline + \
                           _TERM.bold + \
                           (_TERM.red if coeff < 0 else _TERM.green) + \
                           word + \
                           _TERM.normal
            matches = list(re.compile(r'\b' + word + r'\b').finditer(text))
            matches.reverse()
            for match in matches:
                start = match.start()
                text = text[:start] + colored_word + text[start+len(word):]
        return text

    def improve_explanation(self, example, y, explanation):
        class_name = _TERM.bold + \
                     _TERM.color(y) + \
                     self.class_names[y] + \
                     _TERM.normal
        document = self.highlight_words(self.full_documents[example], explanation)

        print(("The model thinks that this document is '{class_name}' " +
              "because of the highlighted words:\n" +
              "\n" +
              "=" * 80 + "\n" +
              "{document}\n" +
              "\n" +
              "=" * 80 + "\n" +
              "The important words are:\n").format(**locals()))

        for word, coeff in explanation.as_list():
            color = _TERM.red if coeff < 0 else _TERM.green
            coeff = _TERM.bold + color + '{:+3.1f}'.format(coeff) + _TERM.normal
            word = _TERM.bold + word + _TERM.normal
            print('  {:24s} {}'.format(word, coeff))

        # TODO acquire improved explanation

        return explanation

    def get_explanation_perf(self, true_explanation, pred_explanation):
        matches = 0
        for true_word, true_coeff in true_explanation.as_list():
            for pred_word, pred_coeff in pred_explanation.as_list():
                if true_word == pred_word and \
                    np.sign(true_coeff) == np.sign(pred_coeff):
                    matches += 1
        return matches / len(true_explanation.as_list())
