# !/usr/bin/python
# -*- coding: utf-8 -*-
"""Code to load data for experiments.
"""
import urllib
import os
import gzip
import struct
import array
import autograd.numpy as np


def download(url, filename):
    """Download data from a url to a file.

    :param url: The url to download the data from.
    :param filename: The path to download the data to.
    :return: None
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urllib.urlretrieve(url, out_file)


def mnist():
    """Download the MNIST data set.

    :return: A tuple of (train_images, train_labels, test_images, test_labels).
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(local_filename):
        """Parse the labels for a file.

        :param local_filename: A string of the file to parse the labels for.
        :return: An array of the parsed labels.
        """
        with gzip.open(local_filename) as fh:
            _, _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(local_filename):
        """Parse the images for a file.

        :param local_filename: A string of the file to parse the images for.
        :return: An array of the parsed images.
        """
        with gzip.open(local_filename) as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist():
    """Prepare the MNIST data for a neural network, by normalizing the images and giving one hot encodings.

    :return: A tuple of (N_data, train_images, train_labels, test_images, test_labels)
    """
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    n_data = train_images.shape[0]

    return n_data, train_images, train_labels, test_images, test_labels
