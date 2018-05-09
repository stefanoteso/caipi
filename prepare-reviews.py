#!/usr/bin/env python3

import re
import pickle
import numpy as np
#import spacy
from os import listdir
from os.path import join
from collections import Counter

np.set_printoptions(threshold=np.nan)

# Make sure to download the dataset from:
#
#  http://cs.jhu.edu/~ozaidan/rationales
#
# and uncompress it in data/review_polarity_rationales/


#SPACY = spacy.load('en')
#
#TAGS = set(['ADJ', 'NOUN', 'VERB'])
#
#
#def simplify(line):
#    return ' '.join([token.lemma_ for token in SPACY(line)
#                     if token.pos_ in TAGS and token.lemma_ != '-PRON-'])
#
#
#def simplify_rats(rats):
#    words = set()
#    for r in rats:
#        words.update(set(simplify(r).split()))
#    return words


POS_RE = re.compile('<POS> (?P<rationale>[^<>]*) </POS>')
NEG_RE = re.compile('<NEG> (?P<rationale>[^<>]*) </NEG>')
TAGS = {'<POS>', '</POS>', '<NEG>', '</NEG>'}


def read_docs(base_path, label):
    docs = {}
    for rel_path in sorted(listdir(base_path)):
        n = rel_path.split('_')[-1].split('.')[0]
        with open(join(base_path, rel_path), encoding='latin-1') as fp:
            docs[(n, label)] = fp.read().strip()
    return docs


def clean(line, vocab):
    return ' '.join([word for word in line.split() if word in vocab])


def find_rats(line, vocab, regex):
    matches = list(regex.finditer(line))
    if len(matches) == 0:
        return line, None

    ranges = []
    for match in matches:
        ranges.extend([(match.start(), True), (match.end(), False)])
    if not ranges[0] == (0, True):
        ranges = [(0, False)] + ranges
    if not ranges[-1] == (len(line), False):
        ranges = ranges + [(len(line), False)]

    words = clean(line, vocab).split()
    mask = np.zeros((len(matches), len(words)))

    n_words, all_words, j = 0, [], 0
    for i in range(len(ranges) - 1):
        s, is_rat = ranges[i]
        e, _ = ranges[i + 1]
        segment = line[s:e].strip()
        words = [w for w in  segment.split() if not w in TAGS]
        #print(segment, '|', words[n_words:n_words+len(words)])
        if is_rat:
            mask[j, n_words:n_words+len(words)] = 1
            #print(mask[j])
        n_words += len(words)
        all_words.extend(words)
        if is_rat:
            j += 1

    return ' '.join(all_words), mask


def read_docs_with_rats(base_path, label, vocab):
    regex = POS_RE if label >= 0 else NEG_RE

    docs, rats = {}, {}
    rel_paths = sorted(listdir(base_path))
    for k, rel_path in enumerate(rel_paths):
        print('processing {}/{}'.format(k + 1, len(rel_paths)))
        n = rel_path.split('_')[-1].split('.')[0]
        with open(join(base_path, rel_path), encoding='latin-1') as fp:
            line = fp.read().strip()
            line = clean(line, vocab | TAGS)
            line, mask = find_rats(line, vocab, regex)
            docs[(n, label)] = line
            rats[(n, label)] = mask
    return docs, rats


print('Reading documents...')
pos_docs = read_docs(join('data', 'review_polarity_rationales', 'noRats_pos'), +1)
neg_docs = read_docs(join('data', 'review_polarity_rationales', 'noRats_neg'), -1)
docs = {**pos_docs, **neg_docs}

words = []
for text in docs.values():
    words.extend(text.split())
counts = Counter(words)
vocab = {word for word in set(words) if counts[word] >= 4}

# NOTE: check that the count matches the original paper
assert(len(vocab) == 17744)

pos_docs_with_rats, pos_rats = read_docs_with_rats(join('data', 'review_polarity_rationales', 'withRats_pos'), +1, vocab)
neg_docs_with_rats, neg_rats = read_docs_with_rats(join('data', 'review_polarity_rationales', 'withRats_neg'), -1, vocab)

print('stuffing...')

y = np.array([+1] * 1000 + [-1] * 1000)

docs = [None] * (len(pos_docs) + len(neg_docs))
rats = [None] * (len(pos_docs) + len(neg_docs))
for (n, label), doc in pos_docs.items():
    docs[int(n)] = clean(doc, vocab)
for (n, label), doc in neg_docs.items():
    docs[1000+int(n)] = clean(doc, vocab)
for (n, label), doc in pos_docs_with_rats.items():
    docs[int(n)] = doc
for (n, label), doc in neg_docs_with_rats.items():
    docs[1000+int(n)] = doc
for (n, label), masks in pos_rats.items():
    rats[int(n)] = masks
for (n, label), masks in neg_rats.items():
    rats[1000+int(n)] = -masks

print('saving...')

dataset = {
    'y': y,
    'docs': docs,
    'explanations': rats,
}

with open(join('data', 'review_polarity_rationales_new.pickle'), 'wb') as fp:
    pickle.dump(dataset, fp)
