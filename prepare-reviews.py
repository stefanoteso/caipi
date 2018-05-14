#!/usr/bin/env python3

import re
import pickle
import numpy as np
import spacy
from os import listdir
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from caipi import load, dump


N_DOCUMENTS_PER_CLASS = 10
METHOD = 'global'


# Make sure to download the dataset from:
#
#  http://cs.jhu.edu/~ozaidan/rationales
#
# and uncompress it in data/review_polarity_rationales/


SPACY = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


POS_TAGS = {'ADJ', 'ADV', 'NOUN', 'VERB'}
RAT_TAGS = {'POS', '/POS', 'NEG', '/NEG'}
RAT_TAGS2 = {'<' + tag + '>' for tag in RAT_TAGS}
REGEX = re.compile('<(POS|NEG)> (?P<rationale>[^<>]*) </(POS|NEG)>')


# NOTE 'oscar winner'
# XXX what about negation? it would need POS tagging maybe


def simplify(line):
    tokens = SPACY(line)
    valid_lemmas = []
    for i, token in enumerate(tokens):
        if (token.pos_ in POS_TAGS and
            token.lemma_ != '-PRON-'):
            valid_lemmas.append(token.lemma_)
        if (token.text in RAT_TAGS and
            tokens[i-1].text == '<' and
            tokens[i+1].text == '>'):
            valid_lemmas.append('<' + token.text + '>')
    return ' '.join(valid_lemmas)


def process_rats(line):
    matches = list(REGEX.finditer(line))
    if len(matches) == 0:
        return line, None

    ranges = []
    for match in matches:
        ranges.extend([(match.start(), True), (match.end(), False)])
    if not ranges[0] == (0, True):
        ranges = [(0, False)] + ranges
    if not ranges[-1] == (len(line), False):
        ranges = ranges + [(len(line), False)]

    words = line.split()
    masks = np.zeros((len(matches), len(words)))

    j = 0
    valid_words = []
    for i in range(len(ranges) - 1):
        s, is_rationale = ranges[i]
        e, _ = ranges[i + 1]
        segment_words = [word for word in line[s:e].strip().split()
                         if not word in RAT_TAGS2]

        if is_rationale:
            masks[j, len(valid_words):len(valid_words)+len(segment_words)] = 1
            j += 1
        valid_words.extend(segment_words)

    return ' '.join(valid_words), masks


def read_docs(base_path, label):
    docs, rats = [], []
    rel_paths = sorted(listdir(base_path))
    for k, rel_path in enumerate(rel_paths):
        if k >= N_DOCUMENTS_PER_CLASS:
            break
        print('processing {}/{} {}'.format(k + 1, len(rel_paths), rel_path))
        n = rel_path.split('_')[-1].split('.')[0]
        with open(join(base_path, rel_path), encoding='latin-1') as fp:
            doc = simplify(fp.read().strip())
            doc, masks = process_rats(doc)
            docs.append(doc)
            rats.append(masks)
    return docs, rats


np.set_printoptions(threshold=np.nan)

try:
    print('Loading...')
    y, docs, rats = load('reviews.pickle')

except:
    print('Reading documents...')
    pos_docs, pos_rats = read_docs(join('data', 'review_polarity_rationales', 'withRats_pos'), +1)
    neg_docs, neg_rats = read_docs(join('data', 'review_polarity_rationales', 'withRats_neg'), -1)

    print('Saving...')
    y = np.array([+1] * len(pos_docs) + [-1] * len(neg_docs))
    docs = pos_docs + neg_docs
    rats = pos_rats + neg_rats
    dump('reviews.pickle', (y, docs, rats))

vectorizer = TfidfVectorizer(lowercase=False)
X = vectorizer.fit_transform(docs).toarray()
vocabulary = np.array(vectorizer.get_feature_names())

model = SGDClassifier(penalty='l1', random_state=0).fit(X, y)
coef = np.abs(model.coef_.ravel())
selected = [i for i in coef.argsort()[::-1]
            if coef[i] >= 1e-9]
relevant_words = set(vocabulary[selected])

print('feature selector acc =', model.score(X, y))
print('# words =', len(vocabulary))
print('# relevant words =', len(relevant_words))

rats = []
for doc in docs:

    words = doc.split()
    relevant_indices = [i for i in range(len(words))
                        if words[i] in relevant_words]
    print('# relevant in doc =', len(relevant_indices))
    mask = np.zeros((1, len(words)))
    mask[0, relevant_indices] = 1
    rats.append(mask)

dataset = {
    'y': y,
    'docs': docs,
    'explanations': rats,
}

with open(join('data', 'review_polarity_rationales.pickle'), 'wb') as fp:
    pickle.dump(dataset, fp)
