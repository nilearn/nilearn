"""Utilities for testing nisl.
"""
# Author: Alexandre Abrahame, Philippe Gervais
# License: simplified BSD
import os
import urllib2

import numpy as np
import scipy.signal

from . import datasets
from . import masking


class mock_urllib2(object):

    def __init__(self):
        """Object that mocks the urllib2 module to store downloaded filenames.

        `downloaded_files` is the list of the files whose download has been
        requested.
        """
        self.urls = []

    class HTTPError(urllib2.URLError):
        code = 404

    class URLError(urllib2.URLError):
        pass

    def urlopen(self, url):
        self.urls.append(url)
        return url

    def reset(self):
        self.urls = []


def mock_chunk_read_(response, local_file, initial_size=0, chunk_size=8192,
                     report_hook=None, verbose=0):
    return


def mock_chunk_read_raise_error_(response, local_file, initial_size=0,
                                 chunk_size=8192, report_hook=None,
                                 verbose=0):
    raise urllib2.HTTPError("url", 418, "I'm a teapot", None, None)


def mock_uncompress_file(file, delete_archive=True):
    return


def mock_get_dataset(dataset_name, file_names, data_dir=None, folder=None):
    """ Mock the original _get_dataset function

    For test purposes, this function acts as a two-pass function. During the
    first run (normally, the fetching function is checking if the dataset
    already exists), the function will throw an error and create the files
    to prepare the second pass. After this first call, any other call will
    succeed as the files have been created.

    This behavior is made to force downloading of the dataset.
    """
    data_dir = datasets._get_dataset_dir(dataset_name, data_dir=data_dir)
    if not (folder is None):
        data_dir = os.path.join(data_dir, folder)
    file_paths = []
    error = None
    for file_name in file_names:
        full_name = os.path.join(data_dir, file_name)
        if not os.path.exists(full_name):
            error = IOError("No such file: '%s'" % full_name)
            dirname = os.path.dirname(full_name)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            open(full_name, 'w').close()
        file_paths.append(full_name)
    if error is not None:
        raise error
    return file_paths


def generate_timeseries(n_instants, n_features,
                        rand_gen=None):
    """Generate some random timeseries. """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)
    # TODO: add an "order" keyword
    return rand_gen.randn(n_instants, n_features)


def generate_regions_ts(n_features, n_regions,
                        overlap=0,
                        rand_gen=None,
                        window="boxcar"):
    """Generate some regions.

    Parameters
    ==========
    overlap (int)
        Number of overlapping voxels between two regions (more or less)
    window (str)
        Name of a window in scipy.signal. e.g. "hamming".

    Returns
    =======
    regions (numpy.ndarray)
        timeseries representing regions.
        shape (n_features, n_regions)
    """

    if rand_gen is None:
        rand_gen = np.random.RandomState(0)
    if window is None:
        window = "boxcar"

    assert(n_features > n_regions)

    # Compute region boundaries indices.
    # Start at 1 to avoid getting an empty region
    boundaries = np.zeros(n_regions + 1)
    boundaries[-1] = n_features
    boundaries[1:-1] = rand_gen.permutation(range(1, n_features)
                                            )[:n_regions - 1]
    boundaries.sort()

    regions = np.zeros((n_regions, n_features), order="C")
    overlap_end = int((overlap + 1) / 2)
    overlap_start = int(overlap / 2)
    for n in xrange(len(boundaries) - 1):
        start = max(0, boundaries[n] - overlap_start)
        end = min(n_features, boundaries[n + 1] + overlap_end)
        win = scipy.signal.get_window(window, end - start)
        win /= win.mean()  # unity mean
        regions[n, start:end] = win

    return regions


def generate_labeled_regions(shape, n_regions, rand_gen=None, labels=None):
    """Generate a 3D volume with labeled regions.

    Parameters
    ==========
    shape (tuple)
        shape of returned array
    n_regions (integer)
        number of regions to generate. By default (if "labels" is None),
        add a background with value zero.
    labels (iterable)
        labels to use for each zone. If provided, n_regions is unused.
    rand_gen (numpy.random.RandomState object)
        random generator to use for generation.

    Returns
    =======
    regions (numpy.ndarray)
        array of shape "shape", containing region labels.
    """
    n_voxels = shape[0] * shape[1] * shape[2]
    if labels is None:
        labels = xrange(0, n_regions + 1)
        n_regions += 1
    else:
        n_regions = len(labels)

    regions = generate_regions_ts(n_voxels, n_regions, rand_gen=rand_gen)
    # replace weights with labels
    for n, row in zip(labels, regions):
        row[row > 0] = n
    return masking.unmask(regions.sum(axis=0), np.ones(shape, dtype=np.bool))
