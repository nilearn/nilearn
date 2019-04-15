"""
Data generation utilities
"""

import numpy as np
import scipy.signal

from sklearn.utils import check_random_state
import scipy.linalg
import nibabel

from .. import masking
from . import logger


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
    ----------
    overlap: int
        Number of overlapping voxels between two regions (more or less)
    window: str
        Name of a window in scipy.signal. e.g. "hamming".

    Returns
    -------
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
    boundaries[1:-1] = rand_gen.permutation(np.arange(1, n_features)
                                            )[:n_regions - 1]
    boundaries.sort()

    regions = np.zeros((n_regions, n_features), order="C")
    overlap_end = int((overlap + 1) / 2.)
    overlap_start = int(overlap / 2.)
    for n in range(len(boundaries) - 1):
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
    ----------
    n_regions: int
        number of regions to generate

    overlap: int
        approximate number of voxels common to two neighboring regions

    window: str
        name of a window in scipy.signal. Used to get non-uniform regions.

    border: int
        number of background voxels on each side of the 3D volumes.

    Returns
    -------
    maps: nibabel.Nifti1Image
        4D array, containing maps.
    """

    mask = np.zeros(shape, dtype=np.int8)
    mask[border:-border, border:-border, border:-border] = 1
    ts = generate_regions_ts(mask.sum(), n_regions, overlap=overlap,
                             rand_gen=rand_gen, window=window)
    mask_img = nibabel.Nifti1Image(mask, affine)
    return masking.unmask(ts, mask_img), mask_img


def generate_labeled_regions(shape, n_regions, rand_gen=None, labels=None,
                             affine=np.eye(4), dtype=np.int):
    """Generate a 3D volume with labeled regions.

    Parameters
    ----------
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
    -------
    regions: nibabel.Nifti1Image
        data has shape "shape", containing region labels.
    """
    n_voxels = shape[0] * shape[1] * shape[2]
    if labels is None:
        labels = range(0, n_regions + 1)
        n_regions += 1
    else:
        n_regions = len(labels)

    regions = generate_regions_ts(n_voxels, n_regions, rand_gen=rand_gen)
    # replace weights with labels
    for n, row in zip(labels, regions):
        row[row > 0] = n
    data = np.zeros(shape, dtype=dtype)
    data[np.ones(shape, dtype=np.bool)] = regions.sum(axis=0).T
    return nibabel.Nifti1Image(data, affine)


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
    return nibabel.Nifti1Image(data, affine)


def generate_fake_fmri(shape=(10, 11, 12), length=17, kind="noise",
                       affine=np.eye(4), n_blocks=None, block_size=None,
                       block_type='classification',
                       rand_gen=np.random.RandomState(0)):
    """Generate a signal which can be used for testing.

    The return value is a 4D array, representing 3D volumes along time.
    Only the voxels in the center are non-zero, to mimic the presence of
    brain voxels in real signals. Setting n_blocks to an integer generates
    condition blocks, the remaining of the timeseries corresponding
    to 'rest' or 'baseline' condition.

    Parameters
    ----------
    shape: tuple, optional
        Shape of 3D volume

    length: int, optional
        Number of time instants

    kind: string, optional
        Kind of signal used as timeseries.
        "noise": uniformly sampled values in [0..255]
        "step": 0.5 for the first half then 1.

    affine: numpy.ndarray
        Affine of returned images

    n_blocks: int or None
        Number of condition blocks.

    block_size: int or None
        Number of timepoints in a block. Used only if n_blocks is not
        None. Defaults to 3 if n_blocks is not None.

    block_type: str
        Defines if the returned target should be used for
        'classification' or 'regression'.

    Returns
    -------
    fmri: nibabel.Nifti1Image
        fake fmri signal.
        shape: shape + (length,)

    mask: nibabel.Nifti1Image
        mask giving non-zero voxels

    target: numpy.ndarray
        Classification or regression target. Shape of number of
        time points (length). Returned only if n_blocks is not None
    """
    full_shape = shape + (length, )
    fmri = np.zeros(full_shape)
    # Fill central voxels timeseries with random signals
    width = [s // 2 for s in shape]
    shift = [s // 4 for s in shape]

    if kind == "noise":
        signals = rand_gen.randint(256, size=(width + [length]))
    elif kind == "step":
        signals = np.ones(width + [length])
        signals[..., :length // 2] = 0.5
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

    if n_blocks is None:
        return (nibabel.Nifti1Image(fmri, affine),
                nibabel.Nifti1Image(mask, affine))

    block_size = 3 if block_size is None else block_size
    flat_fmri = fmri[mask.astype(np.bool)]
    flat_fmri /= np.abs(flat_fmri).max()
    target = np.zeros(length, dtype=np.int)
    rest_max_size = (length - (n_blocks * block_size)) // n_blocks
    if rest_max_size < 0:
        raise ValueError(
            '%s is too small '
            'to put %s blocks of size %s' % (
                length, n_blocks, block_size))
    t_start = 0
    if rest_max_size > 0:
        t_start = rand_gen.random_integers(0, rest_max_size, 1)[0]
    for block in range(n_blocks):
        if block_type == 'classification':
            # Select a random voxel and add some signal to the background
            voxel_idx = rand_gen.randint(0, flat_fmri.shape[0], 1)[0]
            trials_effect = (rand_gen.random_sample(block_size) + 1) * 3.
        else:
            # Select the voxel in the image center and add some signal
            # that increases with each block
            voxel_idx = flat_fmri.shape[0] // 2
            trials_effect = (
                rand_gen.random_sample(block_size) + 1) * block
        t_rest = 0
        if rest_max_size > 0:
            t_rest = rand_gen.random_integers(0, rest_max_size, 1)[0]
        flat_fmri[voxel_idx, t_start:t_start + block_size] += trials_effect
        target[t_start:t_start + block_size] = block + 1
        t_start += t_rest + block_size
    target = target if block_type == 'classification' \
        else target.astype(np.float)
    fmri = np.zeros(fmri.shape)
    fmri[mask.astype(np.bool)] = flat_fmri
    return (nibabel.Nifti1Image(fmri, affine),
            nibabel.Nifti1Image(mask, affine), target)


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
        density=0.1, random_state=0, verbose=0):
    """Generate signals drawn from a sparse Gaussian graphical model.

    Parameters
    ----------
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

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
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
        1. * topology.sum() / (topology.shape[0] ** 2)),
        verbose=verbose)

    # Generate temporal signals
    signals = generate_signals_from_precisions(precisions,
                                               min_n_samples=min_n_samples,
                                               max_n_samples=max_n_samples,
                                               random_state=random_state)
    return signals, precisions, topology

