import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support as prfs
from .problems import Problem
from .utils import TextMod


# TODO active learning plots + discrepancy for newsgroups
# TODO explanation examples at beginning and end of procedure
# TODO define learning from explanation improvements


class NewsgroupsProblem(Problem):
    """Multi-class document classification.

    Part of it is ripped straight from https://marcotcr.github.io/lime
    """
    CATEGORIES = ['alt.atheism', 'soc.religion.christian']

    def __init__(self, oracle=None, rng=None):
        import nltk

        # NOTE let's keep the quotes, they are pretty informative
        dataset = fetch_20newsgroups(subset='all',
                                     categories=self.CATEGORIES,
                                     remove=('headers', 'footers'),
                                     random_state=rng)

        self.examples = list(range(len(dataset.target)))
        self.data = dataset.data
        self.Y = dataset.target

        # TODO distinguish between learner's and explainer's features

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

        self.processed_data = []
        for text in self.data:
            processed_text = ' '.join(token for token, tag
                                      in nltk.pos_tag(nltk.word_tokenize(text))
                                      if tag in VALID_TAGS)
            self.processed_data.append(processed_text)

        vectorizer = TfidfVectorizer(lowercase=False)
        self.vectorizer = vectorizer.fit(self.processed_data)
        self.X = vectorizer.transform(self.processed_data)

        self.explainer = LimeTextExplainer(class_names=self.CATEGORIES,
                                           verbose=True)

    def set_fold(self, train_examples):
        pass

    def evaluate(self, learner, examples):
        return prfs(self.Y[examples],
                    learner.predict(self.X[examples]),
                    average='weighted')[:3]

    def explain(self, learner, example):
        pipeline = make_pipeline(self.vectorizer, learner.model_)
        explanation = self.explainer.explain_instance(self.processed_data[example],
                                                      pipeline.predict_proba,
                                                      num_features=10)
        # TODO extract datapoints, coefficients, intercept, discrepancy
        return explanation, -1

    def improve(self, example, y):
        return self.Y[example]

    @staticmethod
    def highlight_words(text, explanation):
        import re

        for word, coeff in explanation.as_list():
            colored_word = TextMod.UNDERLINE + TextMod.BOLD + \
                           (TextMod.RED if coeff < 0 else TextMod.GREEN) + \
                           word + TextMod.END
            matches = list(re.compile(r'\b' + word + r'\b').finditer(text))
            matches.reverse()
            for match in matches:
                start = match.start()
                text = text[:start] + colored_word + text[start+len(word):]
        return text

    def improve_explanation(self, explainer, example, y, explanation):
        class_color = TextMod.BOLD + TextMod.GREEN if y else TextMod.RED
        class_name = class_color + self.CATEGORIES[y] + TextMod.END

        print('The model thinks that this document:')
        print('-8<-' * 20)
        print(self.highlight_words(self.data[example], explanation))
        print('-8<-' * 20)
        print('is {}, because of these words:'.format(class_name))
        for word, coeff in explanation.as_list():
            color = TextMod.RED if coeff < 0 else TextMod.GREEN
            coeff = TextMod.BOLD + color + '{:+3.1f}'.format(coeff) + TextMod.END
            word = TextMod.BOLD + word + TextMod.END
            print('  {:16s} : {}'.format(word, coeff))

        # TODO acquire improved explanation

        return explanation, -1
