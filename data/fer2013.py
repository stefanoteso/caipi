#!/usr/bin/env python3
"""The facial emotion recognition dataset can be downloaded at:

    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

print('loading')
y, images = [], []
with open('fer2013.csv', 'rt') as fp:
    for line in fp.readlines()[1:]:
        label, pixels, _ = line.split(',')
        y.append(label)
        image = np.array(pixels.split()).astype(np.uint8).reshape(48, 48)
        images.append(image)

print('reshaping')
X = np.array(images)
y = np.array(y).astype(np.uint8)

print('saving')
with open('fer2013.pickle', 'wb') as fp:
    pickle.dump({
            'data': X,
            'target'i: y,
            'class_names': ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral'),
        }, fp, protocol=pickle.HIGHEST_PROTOCOL)

# print('drawing')
# for i, x in enumerate(X):
#     fig, ax = plt.subplots(1, 1)
#     fig.set_size_inches((1, 1))
#     ax.imshow(x, cmap='gist_gray', aspect='equal')
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     fig.savefig('example-{}.png'.format(i), bbox_inches='tight', pad_inches=0)
