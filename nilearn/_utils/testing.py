"""Utilities for testing nilearn.
"""
# Author: Alexandre Abrahame, Philippe Gervais
# License: simplified BSD
import os
import sys
import urllib2
import contextlib
import warnings

import numpy as np
import scipy.signal
from sklearn.utils import check_random_state
import scipy.linalg

from nibabel import Nifti1Image
import nibabel

from .. import datasets
from .. import masking
from . import logger


@contextlib.contextmanager
def write_tmp_imgs(*imgs, **kwargs):
    """Context manager for writing Nifti images.

    Write nifti images in a temporary location, and remove them at the end of
    the block.

    Parameters
    ==========
    imgs: Nifti1Image
        Several Nifti images. Every format understood by nibabel.save is
        accepted.

    create_files: bool
        if True, imgs are written on disk and filenames are returned. If
        False, nothing is written, and imgs is returned as output. This is
        useful to test the two cases (filename / Nifti1Image) in the same
        loop.

    Returns
    =======
    filenames: string or list of
        filename(s) where input images have been written. If a single image
        has been given as input, a single string is returned. Otherwise, a
        list of string is returned.
    """
    valid_keys = set(("create_files",))
    input_keys = set(kwargs.keys())
    invalid_keys = input_keys - valid_keys
    if len(invalid_keys) > 0:
        raise TypeError("%s: unexpected keyword argument(s): %s" %
                        (sys._getframe().f_code.co_name,
                        " ".join(invalid_keys)))
    create_files = kwargs.get("create_files", True)

    if create_files:
        filenames = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for img in imgs:
                filename = os.tempnam(None, "nilearn_") + ".nii"
                filenames.append(filename)
                nibabel.save(img, filename)

        if len(imgs) == 1:
            yield filenames[0]
        else:
            yield filenames

        for filename in filenames:
            os.remove(filename)
    else:  # No-op
        if len(imgs) == 1:
            yield imgs[0]
        else:
            yield imgs


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
    overlap: int
        Number of overlapping voxels between two regions (more or less)
    window: str
        Name of a window in scipy.signal. e.g. "hamming".

    Returns
    =======
    regions: numpy.ndarray
        regions, nepresented as signals.
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
        start = int(max(0, boundaries[n] - overlap_start))
        end = int(min(n_features, boundaries[n + 1] + overlap_end))
        win = scipy.signal.get_window(window, end - start)
        win /= win.mean()  # unity mean
        regions[n, start:end] = win

    return regions


def generate_maps(shape, n_regions, overlap=0, border=1,
                  window="boxcar", rand_gen=None, affine=np.eye(4)):
    """Generate a 4D volume containing several maps.
    Parameters
    ==========
    n_regions: int
        number of regions to generate

    overlap: int
        approximate number of voxels common to two neighboring regions

    window: str
        name of a window in scipy.signal. Used to get non-uniform regions.

    border: int
        number of background voxels on each side of the 3D volumes.

    Returns
    =======
    maps: nibabel.Nifti1Image
        4D array, containing maps.
    """

    mask = np.zeros(shape, dtype=np.int8)
    mask[border:-border, border:-border, border:-border] = 1
    ts = generate_regions_ts(mask.sum(), n_regions, overlap=overlap,
                             rand_gen=rand_gen, window=window)
    mask_img = Nifti1Image(mask, affine)
    return masking.unmask(ts, mask_img), mask_img


def generate_labeled_regions(shape, n_regions, rand_gen=None, labels=None,
                             affine=np.eye(4), dtype=np.int):
    """Generate a 3D volume with labeled regions.

    Parameters
    ==========
    shape: tuple
        shape of returned array

    n_regions: int
        number of regions to generate. By default (if "labels" is None),
        add a background with value zero.

    labels: iterable
        labels to use for each zone. If provided, n_regions is unused.

    rand_gen: numpy.random.RandomState
        random generator to use for generation.

    affine: numpy.ndarray
        affine of returned image

    Returns
    =======
    regions: nibabel.Nifti1Image
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
    data = np.zeros(shape, dtype=dtype)
    data[np.ones(shape, dtype=np.bool)] = regions.sum(axis=0).T
    return Nifti1Image(data, affine)


def generate_labeled_regions_large(shape, n_regions, rand_gen=None,
                                   affine=np.eye(4)):
    """Similar to generate_labeled_regions, but suitable for a large number of
    regions.

    See generate_labeled_regions for details.
    """
    if rand_gen is None:
        rand_gen = np.random.RandomState(0)
    data = rand_gen.randint(n_regions + 1, size=shape)
    if len(np.unique(data)) != n_regions + 1:
        raise ValueError("Some labels are missing. Maybe shape is too small.")
    return Nifti1Image(data, affine)


def generate_fake_fmri(shape=(10, 11, 12), length=17, kind="noise",
                       affine=np.eye(4), rand_gen = np.random.RandomState(0)):
    """Generate a signal which can be used for testing.

    The return value is a 4D array, representing 3D volumes along time.
    Only the voxels in the center are non-zero, to mimic the presence of
    brain voxels in real signals.

    Parameters
    ==========
    shape: tuple, optional
        Shape of 3D volume

    length: int, optional
        Number of time instants

    kind: string, optional
        Kind of signal used as timeseries.
        "noise": uniformly sampled values in [0..255]
        "step": 0.5 for the first half then 1.

    affine: numpy.ndarray
        affine of returned images

    Returns
    =======
    fmri: nibabel.Nifti1Image
        fake fmri signal.
        shape: shape + (length,)

    mask: nibabel.Nifti1Image
        mask giving non-zero voxels
    """
    full_shape = shape + (length, )
    fmri = np.zeros(full_shape)
    # Fill central voxels timeseries with random signals
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


def is_spd(M, decimal=15):
    """Assert that input matrix is symmetric positive definite.

    M must be symmetric down to specified decimal places.
    The check is performed by checking that all eigenvalues are positive.

    Parameters
    ==========
    M: numpy.ndarray
        symmetric positive definite matrix.

    Returns
    =======
    answer: boolean
        True if matrix is symmetric positive definite, False otherwise.
    """
    if not np.allclose(M, M.T, atol=0.1 ** decimal):
        print("matrix not symmetric to %d decimals" % decimal)
        return False
    eigvalsh = np.linalg.eigvalsh(M)
    ispd = eigvalsh.min() > 0
    if not ispd:
        print("matrix has a negative eigenvalue: %.3f" % eigvalsh.min())
    return ispd


def generate_signals_from_precisions(precisions,
                                     min_n_samples=50, max_n_samples=100,
                                     random_state=0):
    """Generate timeseries according to some given precision matrices.

    Signals all have zero mean.

    Parameters
    ----------
    precisions: list of numpy.ndarray
        list of precision matrices. Every matrix must be square (with the same
        size) and positive definite. The output of
        generate_group_sparse_gaussian_graphs() can be used here.

    min_samples, max_samples: int
        the number of samples drawn for each timeseries is taken at random
        between these two numbers.

    Returns
    -------
    signals: list of numpy.ndarray
        output signals. signals[n] corresponds to precisions[n], and has shape
        (sample number, precisions[n].shape[0]).
    """
    random_state = check_random_state(random_state)

    signals = []
    n_samples = random_state.randint(min_n_samples, high=max_n_samples,
                                     size=len(precisions))

    mean = np.zeros(precisions[0].shape[0])
    for n, prec in zip(n_samples, precisions):
        signals.append(random_state.multivariate_normal(mean,
                                                    np.linalg.inv(prec),
                                                    (n,)))
    return signals


def generate_group_sparse_gaussian_graphs(
        n_subjects=5, n_features=30, min_n_samples=30, max_n_samples=50,
        density=0.1, random_state=0):
    """Generate signals drawn from a sparse Gaussian graphical model.

    Parameters
    ==========
    n_subjects : int, optional
        number of subjects

    n_features : int, optional
        number of signals per subject to generate

    density : float, optional
        density of edges in graph topology

    min_n_samples, max_n_samples : int, optional
        Each subject have a different number of samples, between these two
        numbers. All signals for a given subject have the same number of
        samples.

    random_state : int or numpy.random.RandomState instance, optional
        random number generator, or seed.

    Returns
    =======
    subjects : list of numpy.ndarray, shape for each (n_samples, n_features)
        subjects[n] is the signals for subject n. They are provided as a numpy
        len(subjects) = n_subjects. n_samples varies according to the subject.

    precisions : list of numpy.ndarray
        precision matrices.

    topology : numpy.ndarray
        binary array giving the graph topology used for generating covariances
        and signals.
    """

    random_state = check_random_state(random_state)
    # Generate topology (upper triangular binary matrix, with zeros on the
    # diagonal)
    topology = np.empty((n_features, n_features))
    topology[:, :] = np.triu((
        random_state.randint(0, high=int(1. / density),
                         size=n_features * n_features)
        ).reshape(n_features, n_features) == 0, k=1)

    # Generate edges weights on topology
    precisions = []
    mask = topology > 0
    for _ in range(n_subjects):

        # See also sklearn.datasets.samples_generator.make_sparse_spd_matrix
        prec = topology.copy()
        prec[mask] = random_state.uniform(low=.1, high=.8, size=(mask.sum()))
        prec += np.eye(prec.shape[0])
        prec = np.dot(prec.T, prec)

        # Assert precision matrix is spd
        np.testing.assert_almost_equal(prec, prec.T)
        eigenvalues = np.linalg.eigvalsh(prec)
        if eigenvalues.min() < 0:
            raise ValueError("Failed generating a positive definite precision "
                             "matrix. Decreasing n_features can help solving "
                             "this problem.")
        precisions.append(prec)

    # Returns the topology matrix of precision matrices.
    topology += np.eye(*topology.shape)
    topology = np.dot(topology.T, topology)
    topology = topology > 0
    assert(np.all(topology == topology.T))
    logger.log("Sparsity: {0:f}".format(
        1. * topology.sum() / (topology.shape[0] ** 2)))

    # Generate temporal signals
    signals = generate_signals_from_precisions(precisions,
                                               min_n_samples=min_n_samples,
                                               max_n_samples=max_n_samples,
                                               random_state=random_state)
    return signals, precisions, topology
