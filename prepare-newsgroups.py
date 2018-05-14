#!/usr/bin/env python3

import numpy as np
import spacy
from os.path import join
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from caipi import load, dump


SPACY = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
POS_TAGS = {'ADJ', 'ADV', 'NOUN', 'VERB'}


def simplify(line):
    tokens = SPACY(line)
    valid_lemmas = []
    for i, token in enumerate(tokens):
        if (token.pos_ in POS_TAGS and
            token.lemma_ != '-PRON-'):
            valid_lemmas.append(token.lemma_)
    return ' '.join(valid_lemmas)


categories = ['alt.atheism', 'soc.religion.christian']
twenty = fetch_20newsgroups(subset='all',
                            categories=categories,
                            remove=['headers', 'footers'],
                            random_state=0)
docs = [simplify(doc) for doc in twenty.data]


vectorizer = TfidfVectorizer(lowercase=False)
X = vectorizer.fit_transform(docs).toarray()
y = twenty.target

vocabulary = np.array(vectorizer.get_feature_names())
feature_selector = SGDClassifier(penalty='l1', random_state=0)
feature_selector.fit(X, y)
print('feature_selector acc =', feature_selector.score(X, y))
coef = np.abs(feature_selector.coef_.ravel())

selected_indices = [i for i in coef.argsort()[::-1] if coef[i] >= 1]
selected_words = vocabulary[selected_indices]

print('# words =', len(vocabulary))
print('# selected words =', len(selected_words))

docs2 = []
rats = []
keep = []
for i, doc in enumerate(docs):
    words = np.array(doc.split())
    if len(words) == 0:
        continue
    indices = [i for i in range(len(words))
               if words[i] in selected_words]
    print('%% relevant =', len(indices) / len(words))
    mask = np.zeros((1, len(words)))
    mask[0, indices] = 1
    print(mask)
    rats.append(mask)
    docs2.append(doc)
    keep.append(i)
y2 = y[keep]

vectorizer = TfidfVectorizer(lowercase=False, vocabulary=selected_words)
X2 = vectorizer.fit_transform(docs2).toarray()
feature_selector = SGDClassifier(penalty='l1', random_state=0)
feature_selector.fit(X2, y2)
print('feature_selector acc =', feature_selector.score(X2, y2))

dataset = {
    'y': y2,
    'docs': docs2,
    'explanations': rats,
}

dump(join('data', 'newsgroups.pickle'), dataset)
