from collections import Counter

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from blessings import Terminal

import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from . import Problem, load, dump


_TERM = Terminal()


class NewsgroupsProblem(Problem):
    """Document classification.

    Partially ripped from https://github.com/marcotcr/lime
    """

    def __init__(self, *args, labels=None, min_words=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_words = min_words

        # TODO use standard 20newsgroups processing, ask Antonio

        from os.path import join

        path = join('data', '20newsgroups.pickle')
        try:
            print('loading 20newsgroups...')
            dataset, all_pp_documents = load(path)
        except:
            print('failed, preprocessing 20newsgroups...')
            # NOTE quotes include headers
            dataset = fetch_20newsgroups(subset='all',
                                         remove=('headers', 'footers', 'quotes'),
                                         random_state=0)
            all_pp_documents = self.preprocess(dataset.data)

            print('caching preprocessed dataset...')
            dump(path, (dataset, all_pp_documents))

        print('\n\nPROCESSED doc\n{}\n\n'.format(all_pp_documents[:20]))
        self.class_names = dataset.target_names
        if labels is None:
            self.labels = list(range(len(self.class_names)))
        else:
            self.labels = [self.class_names.index(label) for label in labels]
        indices = list(np.where(np.isin(dataset.target, self.labels))[0])
        indices = [i for i in indices if len(all_pp_documents[i].split()) >= min_words]

        self.examples = list(range(len(indices)))
        self.y = dataset.target[indices]
        self.documents = [all_pp_documents[i] for i in indices]
        self.vectorizer = TfidfVectorizer(lowercase=False).fit(self.documents)
        self.X = self.vectorizer.transform(self.documents)
        self.full_documents = [dataset.data[i] for i in indices]

    def wrap_preproc(self, model):
        return model

    @staticmethod
    def preprocess(data,
                   tokenizer=None,
                   stopwords=set(stopwords.words('english')),
                   stemmer=None,
                   filter_pos_tag=False,
                   lemmatizer='wordnet',
                   min_df=1, max_df=0.3):
        """
        Small NLP pipeline:
        0. lower casing
        0. tokenize
        1. stopword removal
        2. lemmatize/stemming
        3. filtering by document frequency
        4. pos tagging filtering
        Reduces documents to lists of adjectives, nouns, and verbs.
        """

        n_docs = len(data)
        if isinstance(min_df, float):
            min_df = min_df * n_docs
        if isinstance(max_df, float):
            max_df = max_df * n_docs
        print('Min doc freq {} and Max doc freq {} (all docs {})'.format(min_df, max_df, n_docs))

        def get_stemmer(stemmer='snowball'):

            if stemmer == 'porter':
                return PorterStemmer()
            elif stemmer == 'lancaster':
                return LancasterStemmer()
            elif stemmer == 'snowball':
                return SnowballStemmer('english')
            else:
                raise ValueError('Unrecognized stemmer', stemmer)

        VALID_TAGS = set([
            'FW',   # Foreign word
            'JJ',   # Adjective
            'JJR',  # Adjective, comparative
            'JJS',  # Adjective, superlative
            'NN',   # Noun, singular or mass
            'NNS',  # Noun, plural
            'NNP',  # Proper noun, singular
            'NNPS',  # Proper noun, plural
            'UH',   # Interjection
            'VB',   # Verb, base form
            'VBD',  # Verb, past tense
            'VBG',  # Verb, gerund or present participle
            'VBN',  # Verb, past participle
            'VBP',  # Verb, non-3rd person singular present
            'VBZ',  # Verb, 3rd person singular present
        ])

        # stemmer = PorterStemmer()
        # processed_data = []
        # for i, text in enumerate(data):
        #     print('preprocessing document {} of {}'.format(i, len(data)), end='\t\r')
        #     processed_text = ' '.join(stemmer.stem(token) for token, tag
        #                               in nltk.pos_tag(nltk.word_tokenize(text))
        #                               if tag in VALID_TAGS)
        #     processed_data.append(processed_text)
        # return processed_data

        doc_freq = Counter()
        processed_data = []
        for i, text in enumerate(data):
            print('preprocessing document {} of {}'.format(i, len(data)), end='\t\r')
            # tokens = word_tokenize(text.lower())
            if tokenizer is None:
                tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text.lower())
            if stopwords:
                tokens = [t for t in tokens if t not in stopwords]
            if filter_pos_tag:
                tokens = [t for t, tag in nltk.pos_tag(tokens) if tag in VALID_TAGS]
            if lemmatizer is not None:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(t) for t in tokens]
            elif stemmer is not None:
                stemmer = get_stemmer(stemmer)
                tokens = [stemmer.stem(t) for t in tokens]

            for t in tokens:
                doc_freq[t] += 1

            tokens = [t for t in tokens if min_df <= doc_freq[t] <= max_df]
            processed_data.append(' '.join(t for t in tokens))

        return processed_data

    def wrap_preproc(self, model):
        return model

    def explain(self, learner, train_examples, example, y,
                num_samples=5000, num_features=10, discretize=None):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline

        explainer = LimeTextExplainer(class_names=self.labels)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        pipeline = make_pipeline(self.vectorizer, learner)

        document = self.documents[example]
        explanation = explainer.explain_instance(document,
                                                 pipeline.predict_proba,
                                                 model_regressor=local_model,
                                                 num_features=num_features,
                                                 num_samples=num_samples)
        return explanation

    def improve(self, example, y):
        return self.y[example]

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
                text = text[:start] + colored_word + text[start + len(word):]
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
