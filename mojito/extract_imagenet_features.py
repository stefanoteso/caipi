import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
import os
import sys
import logging
import pickle
import gzip

import numpy
from keras import applications

CNN_MODELS = {'vgg16': applications.VGG16,
              'vgg19': applications.VGG19,
              'resnet50': applications.ResNet50,
              'xception': applications.Xception,
              'inception': applications.InceptionV3}

if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help='Specify the path to the preprocessed image data as numpy array')

    parser.add_argument('-l', '--labels', type=str,
                        default=None,
                        help='Path to the label information (Y).'
                        ' If None, assumed to be a numpy archive "classes.npy in the same dir of data')

    parser.add_argument('-m', '--model', type=str,
                        default='vgg16',
                        help='Deep convolutional model pre trained on imagenet to employ (vgg16|vgg19|resnet50|inception|xception)')

    parser.add_argument('-o', '--output-path', type=str,
                        default='./VOC2012/proc/',
                        help='Output path to extracted features')

    parser.add_argument('-b', '--batch-size', type=int,
                        default=16,
                        help='Batch size to process images at once')

    parser.add_argument('-a', '--aggregate', type=str,
                        default='avg',
                        help='Operation to aggregate last layers (avg|max|flatten)')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    #
    # creating output dirs if they do not exist
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'JPEG'), exist_ok=True)

    #
    # loading (preprocessed) dataset
    # expected as a tensor of n-images X width X height X channels
    data = numpy.load(args.data)
    logging.info('Loaded data from {}\n\tshape: {}'.format(args.data,   data.shape))

    img_rows, img_cols, img_channels = data.shape[1:]

    #
    # loading deep cnn pretrained on imagenet
    imagenet_model = CNN_MODELS[args.model](weights='imagenet',
                                            include_top=False,
                                            input_shape=(img_rows, img_cols, img_channels),
                                            pooling=args.aggregate)

    imagenet_model.summary()
    logging.info('Loaded imagenet model!')

    #
    # getting new data representation
    logging.info('\nExtracting features from data...')
    trans_start_t = perf_counter()
    data_repr = imagenet_model.predict(data, verbose=args.verbose, batch_size=args.batch_size)
    trans_end_t = perf_counter()
    logging.info('\nData transformed from {} to {} in {} secs!'.format(data.shape,
                                                                       data_repr.shape,
                                                                       trans_end_t - trans_start_t))

    #
    # saving to pickle file
    data_outpath = os.path.join(args.output_path, '{}-{}x{}x{}.pklz'.format(args.model,
                                                                            img_rows,
                                                                            img_cols,
                                                                            img_channels))
    #
    # getting labels
    label_path = args.labels
    if label_path is None:
        label_path = os.path.join(os.path.dirname(args.data), 'classes.npy')

    y = numpy.load(label_path)
    logging.info('Loaded classes from {}'.format(label_path))

    with gzip.open(data_outpath, 'wb') as f:
        pickle.dump((data_repr, y), f)
        logging.info('New data repr dumped to {}'.format(data_outpath))
