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

import xml.etree.ElementTree as ET

import numpy
import skimage
import skimage.io
import skimage.transform
# import matplotlib.pyplot as pyplot


def collect_image_paths(dir, ext='.jpg'):
    """
    Load all images of specified extension inside a directory
    (recursive!)
    """
    logging.info('Looking for {} images in dir {}'.format(ext, dir))

    img_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(ext):
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
                logging.debug('\tfound image {}'.format(img_path))

    logging.info('Gathered {} image paths'.format(len(img_paths)))
    return img_paths


# def load_images_as_arrays(paths,
#                           # mode='RGB',
#                           batch
#                           ):
#     """
#     Wrapping scipy's ndimage.imread()
#     """
#     imgs = []
#     for i, p in enumerate(paths[batch[0]:batch[1]]):
#         print('\tLoading img {}/{}'.format(i + batch[0], batch[1]))
#         imgs.append(skimage.io.imread(p))

#     return imgs


# def scale_images(images, scaled_res):
#     imgs = []
#     for i, img in enumerate(images):
#         print('\tscaling img {}/{}'.format(i + batch[0], batch[1]))
#         imgs.append(skimage.transform.resize(img, output_shape=scaled_res))

#     return imgs

VOC_CLASSES = set(['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor', 'ambigious'])

SORTED_VOC_CLASSES = sorted(VOC_CLASSES)
VOC_CLASSES_DICT = {c: i for i, c in enumerate(SORTED_VOC_CLASSES)}


def get_class_from_annotation(xml_file_path):
    xml = ET.parse(xml_file_path)
    return xml.find('object/name').text


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str,
                        help='Specify the path to the images directory')

    parser.add_argument('-a', '--annotations-dir', type=str,
                        help='Annotations dir (e.g., VOCdevkit/VOC2012/Annotations/)')

    parser.add_argument('-e', '--data-exts', type=str,
                        default='.jpg',
                        help='Image extension (default .jpg)')

    # parser.add_argument('-b', '--batch-size', type=int,
    #                     default=100,
    #                     help='Batch size to process images')

    parser.add_argument('-o', '--output-path', type=str,
                        default='./VOC2012/proc/',
                        help='path to processed files')

    parser.add_argument('-r', '--res', type=int, nargs='+',
                        default=None,
                        help='The final image size to rescale images into (e.g. (224, 224)).'
                        ' If not specified (None), no rescaling applied')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')
    #
    # parsing the args
    args = parser.parse_args()

    #
    # creating output dirs if they do not exist
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'JPEG'), exist_ok=True)

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    img_paths = collect_image_paths(dir=args.dir, ext=args.data_exts)

    classes = []
    with open(os.path.join(args.output_path, 'class-names.txt'), 'w') as f:
        f.writelines(['{},{}\n'.format(c, i)
                      for i, c in enumerate(SORTED_VOC_CLASSES)])

    proc_imgs = numpy.zeros((len(img_paths),
                             args.res[0], args.res[1],
                             3), dtype=numpy.float32)
    for i, p in enumerate(img_paths):

        img_name = os.path.basename(p)
        xml_file_path = os.path.join(args.annotations_dir,
                                     img_name.replace(args.data_exts,
                                                      '.xml'))

        img_class = get_class_from_annotation(xml_file_path)
        classes.append(VOC_CLASSES_DICT[img_class])

        #
        # load from disk
        img = skimage.io.imread(p)

        #
        # scaling
        scaled_img = skimage.transform.resize(img, output_shape=args.res)

        proc_imgs[i] = scaled_img

        #
        # saving as jpeg again
        scaled_img_path = os.path.join(args.output_path,
                                       'JPEG',
                                       img_name)
        skimage.io.imsave(scaled_img_path,
                          scaled_img,
                          quality=100)
        print('Processed image {}/{}'.format(i + 1, len(img_paths)),
              # end='\t\r'
              )

    classes = numpy.array(classes)
    classes_file = os.path.join(args.output_path, 'classes')
    numpy.save(classes_file, classes)
    logging.info('Saved class information to file {}'.format(classes_file))

    dataset_file = os.path.join(args.output_path, 'images-{}x{}'.format(args.res[0], args.res[1]))
    numpy.save(dataset_file, proc_imgs)
    logging.info('Saved image information to file {}'.format(dataset_file))

    # for i, batch_start in enumerate(range(0, len(img_paths), args.batch_size)):
    #     batch_end = batch_start + args.batch_size
    #     batch = (batch_start, batch_end)
    #     print("processing batch {} {}".format(i, batch))

    #     imgs_batch = load_images_as_arrays(img_paths, batch=batch)
    #     logging.info('loaded images, printing shapes {}'.format(set([i.shape for i in imgs_batch])))

    #     scaled_imgs_batch = scale_images(imgs_batch, scaled_res=args.res)
    #     print(len(scaled_imgs_batch))
    #     logging.info('scaled images, printing shapes {}'.format(set([i.shape
    #                                                                  for i in scaled_imgs_batch])))

    #     proc_imgs.extend(scaled_imgs_batch)
