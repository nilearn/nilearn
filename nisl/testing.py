"""Utilities for testing nisl.
"""
# Author: Alexandre Abrahame, Philippe Gervais
# License: simplified BSD
import os
import urllib2

import numpy as np
import scipy.signal

from nibabel import Nifti1Image

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
    """Generate some regions as timeseries.

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


def generate_maps(shape, n_regions, overlap=0, border=1,
                  window="boxcar", rand_gen=None):
    """Generate a 4D volume containing several maps.
    Parameters
    ==========
    border (int)
        number of background voxels on each side of the 3D volumes.

    Returns
    =======
    maps (Nifti1Image)
        4D array, containing maps.
    """

    mask = np.zeros(shape, dtype=np.int8)
    mask[border:-border, border:-border, border:-border] = 1
    ts = generate_regions_ts(mask.sum(), n_regions, overlap=overlap,
                             rand_gen=rand_gen, window=window)
    mask_img = Nifti1Image(mask, np.eye(4))
    return masking.unmask(ts, mask_img), mask_img


def generate_labeled_regions(shape, n_regions, rand_gen=None, labels=None,
                             affine=np.eye(4)):
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
    affine (numpy.ndarray)
        affine of returned image

    Returns
    =======
    regions (Nifti1Image)
        data has shape "shape", containing region labels.
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
    return masking.unmask(regions.sum(axis=0),
                          Nifti1Image(np.ones(shape, dtype=np.int8), affine)
                          )


def generate_fake_fmri(shape=(10, 11, 12), length=17, kind="noise",
                       affine=np.eye(4)):
    """Generate a signal which can be used for testing.

    The return value is a 4D array, representing 3D volumes along time.
    Only the voxels in the center are non-zero, to mimic the presence of
    brain voxels in real signals.

    Parameters
    ==========
    shape (tuple, optional)
        Shape of 3D volume
    length (integer, optional)
        Number of time instants
    kind (string, optional)
        Kind of signal used as timeseries.
        "noise": uniformly sampled values in [0..255]
        "step": 0.5 for the first half then 1.
    affine (numpy.ndarray)
        affine of returned images

    Returns
    =======
    fmri (nibabel.Nifti1Image)
        fake fmri signal.
        shape: shape + (length,)
    mask (nibabel.Nifti1Image)
        mask giving non-zero voxels
    """
    full_shape = shape + (length, )
    fmri = np.zeros(full_shape)
    # Fill central voxels timeseries with random signals
    rand_gen = np.random.RandomState(0)
    width = [s / 2 for s in shape]
    shift = [s / 4 for s in shape]

    if kind == "noise":
        signals = rand_gen.randint(256, size=(width + [length]))
    elif kind == "step":
        signals = np.ones(width + [length])
        signals[..., :length / 2] = 0.5
    else:
        raise ValueError("Unhandled value for parameter 'kind'")

    fmri[shift[0]:shift[0] + width[0],
         shift[1]:shift[1] + width[1],
         shift[2]:shift[2] + width[2],
         :] = signals

    mask = np.zeros(shape)
    mask[shift[0]:shift[0] + width[0],
         shift[1]:shift[1] + width[1],
         shift[2]:shift[2] + width[2]] = 1
    return Nifti1Image(fmri, affine), Nifti1Image(mask, affine)
