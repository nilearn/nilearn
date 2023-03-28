"""
Data generation utilities
"""

from __future__ import annotations

import itertools
import json
import string

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.signal
from nibabel import Nifti1Image
from scipy.ndimage import binary_dilation
from sklearn.utils import check_random_state

from nilearn import datasets, image, maskers, masking
from nilearn._utils import as_ndarray, logger
from nilearn.interfaces.bids._utils import _bids_entities, _check_bids_label


def generate_mni_space_img(n_scans=1, res=30, random_state=0, mask_dilation=2):
    """Generate mni space img.

    Parameters
    ----------
    n_scans : :obj:`int`, optional
        Number of scans.
        Default=1.

    res : :obj:`int`, optional
        Desired resolution, in mm, of output images.
        Default=30.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    mask_dilation : :obj:`int`, optional
        The number of times the binary :term:`dilation<Dilation>` is repeated
        on the mask.
        Default=2.

    Returns
    -------
    inverse_img : Niimg-like object
        Image transformed back to mni space.

    mask_img : Niimg-like object
        Generated mask in mni space.

    """
    rand_gen = check_random_state(random_state)
    mask_img = datasets.load_mni152_brain_mask(resolution=res)
    masker = maskers.NiftiMasker(mask_img).fit()
    n_voxels = image.get_data(mask_img).sum()
    data = rand_gen.randn(n_scans, n_voxels)
    if mask_dilation is not None and mask_dilation > 0:
        mask_img = image.new_img_like(
            mask_img,
            binary_dilation(image.get_data(mask_img),
                            iterations=mask_dilation))
    inverse_img = masker.inverse_transform(data)
    return inverse_img, mask_img


def generate_timeseries(n_timepoints, n_features, random_state=0):
    """Generate some random timeseries.

    Parameters
    ----------
    n_timepoints : :obj:`int`
        Number of timepoints

    n_features : :obj:`int`
        Number of features

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    Returns
    -------
    :obj:`numpy.ndarray` of shape (n_timepoints, n_features)
        Generated time series.

    """
    rand_gen = check_random_state(random_state)
    return rand_gen.randn(n_timepoints, n_features)


def generate_regions_ts(n_features,
                        n_regions,
                        overlap=0,
                        random_state=0,
                        window="boxcar"):
    """Generate some regions as timeseries.

    Parameters
    ----------
    n_features : :obj:`int`
        Number of features.

    n_regions : :obj:`int`
        Number of regions.

    overlap : :obj:`int`, optional
        Number of overlapping voxels between two regions (more or less).
        Default=0.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    window : :obj:`str`, optional
        Name of a window in scipy.signal. e.g. "hamming".
        Default='boxcar'.

    Returns
    -------
    regions : :obj:`numpy.ndarray`
        Regions, represented as signals.
        shape (n_features, n_regions)

    """
    rand_gen = check_random_state(random_state)
    if window is None:
        window = "boxcar"

    assert n_features > n_regions

    # Compute region boundaries indices.
    # Start at 1 to avoid getting an empty region
    boundaries = np.zeros(n_regions + 1)
    boundaries[-1] = n_features
    boundaries[1:-1] = rand_gen.permutation(np.arange(
        1, n_features))[:n_regions - 1]
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


def generate_maps(shape,
                  n_regions,
                  overlap=0,
                  border=1,
                  window="boxcar",
                  random_state=0,
                  affine=np.eye(4)):
    """Generate a 4D volume containing several maps.

    Parameters
    ----------
    n_regions : :obj:`int`
        Number of regions to generate.

    overlap : :obj:`int`, optional
        Approximate number of voxels common to two neighboring regions.
        Default=0.

    window : :obj:`str`, optional
        Name of a window in scipy.signal. Used to get non-uniform regions.
        Default='boxcar'.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    affine : :obj:`numpy.ndarray`, optional
        Affine transformation to use.
        Default=np.eye(4).

    border : :obj:`int`, optional
        Number of background voxels on each side of the 3D volumes.
        Default=1.

    Returns
    -------
    maps : Niimg-like object
        4D image object containing the maps.

    mask_img : Niimg-like object
        3D mask giving non-zero voxels.

    """
    mask = np.zeros(shape, dtype=np.int8)
    mask[border:-border, border:-border, border:-border] = 1
    ts = generate_regions_ts(mask.sum(),
                             n_regions,
                             overlap=overlap,
                             random_state=random_state,
                             window=window)
    mask_img = Nifti1Image(mask, affine)
    return masking.unmask(ts, mask_img), mask_img


def generate_labeled_regions(shape,
                             n_regions,
                             random_state=0,
                             labels=None,
                             affine=np.eye(4),
                             dtype="int32"):
    """Generate a 3D volume with labeled regions.

    Parameters
    ----------
    shape : :obj:`tuple`
        Shape of returned array.

    n_regions : :obj:`int`
        Number of regions to generate. By default (if "labels" is None),
        add a background with value zero.
        Default=None.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    labels : iterable, optional
        Labels to use for each zone. If provided, n_regions is unused.

    affine : :obj:`numpy.ndarray`, optional
        Affine of returned image.
        Default=np.eye(4).

    dtype : :obj:`type`, optional
        Data type of image.
        Default='int32'.

    Returns
    -------
    Niimg-like object
        Data has shape "shape", containing region labels.

    """
    n_voxels = shape[0] * shape[1] * shape[2]
    if labels is None:
        labels = range(0, n_regions + 1)
        n_regions += 1
    else:
        n_regions = len(labels)

    regions = generate_regions_ts(n_voxels,
                                  n_regions,
                                  random_state=random_state)
    # replace weights with labels
    for n, row in zip(labels, regions):
        row[row > 0] = n
    data = np.zeros(shape, dtype=dtype)
    data[np.ones(shape, dtype=bool)] = regions.sum(axis=0).T
    return Nifti1Image(data, affine)


def generate_fake_fmri(shape=(10, 11, 12),
                       length=17,
                       kind="noise",
                       affine=np.eye(4),
                       n_blocks=None,
                       block_size=None,
                       block_type='classification',
                       random_state=0):
    """Generate a signal which can be used for testing.

    The return value is a 4D image, representing 3D volumes along time.
    Only the voxels in the center are non-zero, to mimic the presence of
    brain voxels in real signals. Setting n_blocks to an integer generates
    condition blocks, the remaining of the timeseries corresponding
    to 'rest' or 'baseline' condition.

    Parameters
    ----------
    shape : :obj:`tuple`, optional
        Shape of 3D volume.
        Default=(10, 11, 12).

    length : :obj:`int`, optional
        Number of time instants.
        Default=17.

    kind : :obj:`str`, optional
        Kind of signal used as timeseries.
        "noise": uniformly sampled values in [0..255]
        "step": 0.5 for the first half then 1.
        Default='noise'.

    affine : :obj:`numpy.ndarray`, optional
        Affine of returned images.
        Default=np.eye(4).

    n_blocks : :obj:`int` or None, optional
        Number of condition blocks.
        Default=None.

    block_size : :obj:`int` or None, optional
        Number of timepoints in a block. Used only if n_blocks is not
        None. Defaults to 3 if n_blocks is not None.

    block_type : :obj:`str`, optional
        Defines if the returned target should be used for
        'classification' or 'regression'.
        Default='classification'.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    Returns
    -------
    Niimg-like object
        Fake fmri signal.
        shape: shape + (length,)

    Niimg-like object
        Mask giving non-zero voxels.

    target : :obj:`numpy.ndarray`
        Classification or regression target.
        A 1D array with one element for each time point.
        Returned only if ``n_blocks`` is not None.

    """
    full_shape = shape + (length, )
    fmri = np.zeros(full_shape)
    # Fill central voxels timeseries with random signals
    width = [s // 2 for s in shape]
    shift = [s // 4 for s in shape]

    rand_gen = check_random_state(random_state)
    if kind == "noise":
        signals = rand_gen.randint(256, size=(width + [length]))
    elif kind == "step":
        signals = np.ones(width + [length])
        signals[..., :length // 2] = 0.5
    else:
        raise ValueError("Unhandled value for parameter 'kind'")

    fmri[shift[0]:shift[0] + width[0], shift[1]:shift[1] + width[1],
         shift[2]:shift[2] + width[2], :] = signals

    mask = np.zeros(shape)
    mask[shift[0]:shift[0] + width[0], shift[1]:shift[1] + width[1],
         shift[2]:shift[2] + width[2]] = 1

    if n_blocks is None:
        return (Nifti1Image(fmri, affine),
                Nifti1Image(mask, affine))

    block_size = 3 if block_size is None else block_size
    flat_fmri = fmri[mask.astype(bool)]
    flat_fmri /= np.abs(flat_fmri).max()
    target = np.zeros(length, dtype=int)
    rest_max_size = (length - (n_blocks * block_size)) // n_blocks
    if rest_max_size < 0:
        raise ValueError('%s is too small '
                         'to put %s blocks of size %s' %
                         (length, n_blocks, block_size))
    t_start = 0
    if rest_max_size > 0:
        t_start = rand_gen.randint(0, rest_max_size, 1)[0]
    for block in range(n_blocks):
        if block_type == 'classification':
            # Select a random voxel and add some signal to the background
            voxel_idx = rand_gen.randint(0, flat_fmri.shape[0], 1)[0]
            trials_effect = (rand_gen.random_sample(block_size) + 1) * 3.
        else:
            # Select the voxel in the image center and add some signal
            # that increases with each block
            voxel_idx = flat_fmri.shape[0] // 2
            trials_effect = (rand_gen.random_sample(block_size) + 1) * block
        t_rest = 0
        if rest_max_size > 0:
            t_rest = rand_gen.randint(0, rest_max_size, 1)[0]
        flat_fmri[voxel_idx, t_start:t_start + block_size] += trials_effect
        target[t_start:t_start + block_size] = block + 1
        t_start += t_rest + block_size
    target = target if block_type == 'classification' \
        else target.astype(np.float64)
    fmri = np.zeros(fmri.shape)
    fmri[mask.astype(bool)] = flat_fmri
    return (Nifti1Image(fmri, affine),
            Nifti1Image(mask, affine), target)


def generate_fake_fmri_data_and_design(shapes,
                                       rk=3,
                                       affine=np.eye(4),
                                       random_state=0):
    """Generate random :term:`fMRI` time series and design matrices of given
    shapes.

    Parameters
    ----------
    shapes : :obj:`list` of length-4 :obj:`tuple`s of :obj:`int`
        Shapes of the fmri data to be generated.

    rk : :obj:`int`, optional
        Number of columns in the design matrix to be generated.
        Default=3.

    affine : :obj:`numpy.ndarray`, optional
        Affine of returned images. Must be a 4x4 array.
        Default=np.eye(4).

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    Returns
    -------
    mask : Niimg-like object
        3D mask giving non-zero voxels.

    fmri_data : :obj:`list`
        A 4D volume represented by a list of 3D Niimg-like objects.

    design_matrices : :obj:`list`
        A list of pd.DataFrame

    """
    fmri_data = []
    design_matrices = []
    rand_gen = check_random_state(random_state)
    for i, shape in enumerate(shapes):
        data = rand_gen.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        fmri_data.append(Nifti1Image(data, affine))
        columns = rand_gen.choice(list(string.ascii_lowercase),
                                  size=rk,
                                  replace=False)
        design_matrices.append(
            pd.DataFrame(rand_gen.randn(shape[3], rk), columns=columns))
    mask = Nifti1Image((rand_gen.rand(*shape[:3]) > .5).astype(np.int8),
                       affine)
    return mask, fmri_data, design_matrices


def write_fake_fmri_data_and_design(shapes,
                                    rk=3,
                                    affine=np.eye(4),
                                    random_state=0):
    """Generate random :term:`fMRI` data and design matrices and write them to
    disk.

    Parameters
    ----------
    shapes : :obj:`list` of :obj:`tuple`s of :obj:`int`
        A list of shapes in tuple format.

    rk : :obj:`int`, optional
        Number of columns in the design matrix to be generated.
        Default=3.

    affine : :obj:`numpy.ndarray`, optional
        Affine of returned images.
        Default=np.eye(4).

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    Returns
    -------
    mask_file : :obj:`str`
        Path to mask file.

    fmri_files : :obj:`list` of :obj:`str`
        A list of paths to the generated fmri files.

    design_files : :obj:`list` of :obj:`str`
        A list of paths to the generated design matrix files.

    See Also
    --------
    nilearn._utils.data_gen.generate_fake_fmri_data_and_design

    """
    mask_file, fmri_files, design_files = 'mask.nii', [], []
    rand_gen = check_random_state(random_state)
    for i, shape in enumerate(shapes):
        fmri_files.append('fmri_run%d.nii' % i)
        data = rand_gen.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        Nifti1Image(data, affine).to_filename(fmri_files[-1])
        design_files.append('dmtx_%d.csv' % i)
        pd.DataFrame(rand_gen.randn(shape[3], rk),
                     columns=['', '', '']).to_csv(design_files[-1])
    Nifti1Image((rand_gen.rand(*shape[:3]) > .5).astype(np.int8),
                affine).to_filename(mask_file)
    return mask_file, fmri_files, design_files


def write_fake_bold_img(file_path,
                        shape,
                        affine=np.eye(4),
                        random_state=0):
    """Generate a random image of given shape and write it to disk.

    Parameters
    ----------
    file_path : :obj:`str`
        Output file path.

    shape : :obj:`tuple` of :obj:`int`
        Shape of output array. Should be at least 3D.

    affine : :obj:`numpy.ndarray`, optional
        Affine of returned images.
        Default=np.eye(4).

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    Returns
    -------
    file_path : :obj:`str`
        Output file path.

    """
    rand_gen = check_random_state(random_state)
    data = rand_gen.randn(*shape)
    data[1:-1, 1:-1, 1:-1] += 100
    Nifti1Image(data, affine).to_filename(file_path)
    return file_path


def generate_signals_from_precisions(precisions,
                                     min_n_samples=50,
                                     max_n_samples=100,
                                     random_state=0):
    """Generate timeseries according to some given precision matrices.

    Signals all have zero mean.

    Parameters
    ----------
    precisions : :obj:`list` of :obj:`numpy.ndarray`
        A list of precision matrices. Every matrix must be square (with the
        same size) and positive definite. The output of
        generate_group_sparse_gaussian_graphs() can be used here.

    min_samples, max_samples : :obj:`int`, optional
        The number of samples drawn for each timeseries is taken at random
        between these two numbers. Defaults are 50 and 100.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    Returns
    -------
    signals : :obj:`list` of :obj:`numpy.ndarray`
        Output signals. signals[n] corresponds to precisions[n], and has shape
        (sample number, precisions[n].shape[0]).

    """
    rand_gen = check_random_state(random_state)

    signals = []
    n_samples = rand_gen.randint(min_n_samples,
                                 high=max_n_samples,
                                 size=len(precisions))

    mean = np.zeros(precisions[0].shape[0])
    for n, prec in zip(n_samples, precisions):
        signals.append(
            rand_gen.multivariate_normal(mean, np.linalg.inv(prec), (n, )))
    return signals


def generate_group_sparse_gaussian_graphs(n_subjects=5,
                                          n_features=30,
                                          min_n_samples=30,
                                          max_n_samples=50,
                                          density=0.1,
                                          random_state=0,
                                          verbose=0):
    """Generate signals drawn from a sparse Gaussian graphical model.

    Parameters
    ----------
    n_subjects : :obj:`int`, optional
        Number of subjects.
        Default=5.

    n_features : :obj:`int`, optional
        Number of signals per subject to generate.
        Default=30.

    min_n_samples, max_n_samples : :obj:`int`, optional
        Each subject have a different number of samples, between these two
        numbers. All signals for a given subject have the same number of
        samples. Defaults are 30 and 50.

    density : :obj:`float`, optional
        Density of edges in graph topology.
        Default=0.1.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    verbose : :obj:`int`, optional
        Verbosity level (0 means no message).
        Default=0.

    Returns
    -------
    signals : :obj:`list` of :obj:`numpy.ndarray`, shape for each \
        (n_samples, n_features) signals[n] is the signals for subject n.
        They are provided as a numpy len(signals) = n_subjects.
        n_samples varies according to the subject.

    precisions : :obj:`list` of :obj:`numpy.ndarray`
        Precision matrices.

    topology : :obj:`numpy.ndarray`
        Binary array giving the graph topology used for generating covariances
        and signals.

    """
    rand_gen = check_random_state(random_state)
    # Generate topology (upper triangular binary matrix, with zeros on the
    # diagonal)
    topology = np.empty((n_features, n_features))
    topology[:, :] = np.triu(
        (rand_gen.randint(0,
                          high=int(1. / density),
                          size=n_features * n_features)).reshape(
                              n_features, n_features) == 0,
        k=1)

    # Generate edges weights on topology
    precisions = []
    mask = topology > 0
    for _ in range(n_subjects):

        # See also sklearn.datasets.samples_generator.make_sparse_spd_matrix
        prec = topology.copy()
        prec[mask] = rand_gen.uniform(low=.1, high=.8, size=(mask.sum()))
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
    assert (np.all(topology == topology.T))
    logger.log(
        "Sparsity: {0:f}".format(1. * topology.sum() / (topology.shape[0]**2)),
        verbose=verbose)

    # Generate temporal signals
    signals = generate_signals_from_precisions(precisions,
                                               min_n_samples=min_n_samples,
                                               max_n_samples=max_n_samples,
                                               random_state=rand_gen)
    return signals, precisions, topology


def basic_paradigm(condition_names_have_spaces=False):
    """Generate basic paradigm.

    Parameters
    ----------
    condition_names_have_spaces : :obj:`bool`, optional
        Check for spaces in condition names.
        Default=False.

    Returns
    -------
    events : pd.DataFrame
        Basic experimental paradigm with events data.

    """
    conditions = ['c 0', 'c 0', 'c 0',
                  'c 1', 'c 1', 'c 1',
                  'c 2', 'c 2', 'c 2']

    if not condition_names_have_spaces:
        conditions = [c.replace(' ', '') for c in conditions]
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    durations = 1 * np.ones(9)
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets,
                           'duration': durations})
    return events


def basic_confounds(length, random_state=0):
    """Generate random motion parameters (3 translation directions, 3 rotation
    directions).

    Parameters
    ----------
    length : :obj:`int`
        Length of basic confounds.

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance, \
                   optional
        Random number generator, or seed.
        Default=0.

    Returns
    -------
    confounds : :obj:`pandas.DataFrame`.
        Basic confounds.
        This DataFrame will have six columns: "RotX", "RotY", "RotZ", "X", "Y",
        and "Z".

    """
    rand_gen = check_random_state(random_state)
    columns = ['RotX', 'RotY', 'RotZ', 'X', 'Y', 'Z']
    data = rand_gen.rand(length, 6)
    confounds = pd.DataFrame(data, columns=columns)
    return confounds


def generate_random_img(
    shape,
    affine=np.eye(4),
    random_state=np.random.RandomState(0),
):
    """Create a random 3D or 4D image with a given shape and affine.

    Parameters
    ----------
    shape : length-3 or length-4 tuple
        The shape of the image being generated.
        The number of elements determines the dimensionality of the image.

    affine : 4x4 numpy.ndarray
        The affine of the image

    random_state : numpy.random.RandomState instance, optional
        random number generator.

    Returns
    -------
    data_img : 3D or 4D niimg
        The data image.

    mask_img : 3D niimg
        The mask image.
    """
    data = random_state.standard_normal(size=shape)
    data_img = Nifti1Image(data, affine)
    if len(shape) == 4:
        mask_data = as_ndarray(data[..., 0] > 0.2, dtype=np.int8)
    else:
        mask_data = as_ndarray(data > 0.2, dtype=np.int8)

    mask_img = Nifti1Image(mask_data, affine)

    return data_img, mask_img


def create_fake_bids_dataset(
    base_dir = Path(),
    n_sub = 10,
    n_ses = 2,
    tasks = ["localizer", "main"],
    n_runs = [1, 3],
    with_derivatives = True,
    with_confounds = True,
    confounds_tag = "desc-confounds_timeseries",
    random_state=0,
    entities = None,
):
    """Create a fake :term:`bids<BIDS>` dataset directory with dummy files.

    Returns fake dataset directory name.

    Parameters
    ----------
    base_dir : :obj:`str` or :obj:`pathlib.Path` (Absolute path). \
        Default=pathlib.Path()
        Absolute directory path in which to create the fake :term:`BIDS`
        dataset dir.

    n_sub : :obj:`int`, Defaults=10
        Number of subjects to be simulated in the dataset.

    n_ses : :obj:`int`, Defaults=2
        Number of sessions to be simulated in the dataset.

        Specifying n_ses=0 will only produce runs and files without the
        optional session field.

    tasks : :obj:`list` of :obj:`str`, Defaults=["localizer", "main"]
        List of tasks to be simulated in the dataset.

    n_runs : :obj:`list` of :obj:`int`, Defaults=[1, 3]
        Number of runs to create, where each element indicates the
        number of runs for the corresponding task.
        The length of this list must match the number of items in ``tasks``.
        Each run creates 100 volumes.
        Files will be generated without run entity
        if a value is equal to 0 or less.

    with_derivatives : :obj:`bool`, Defaults=True
        In the case derivatives are included, they come with two spaces and
        descriptions.
        Spaces are 'MNI' and 'T1w'.
        Descriptions are 'preproc' and :term:`fMRIPrep`.
        Only space 'T1w' include both descriptions.

    with_confounds : :obj:`bool`, Defaults=True
        Whether to generate associated confounds files or not.

    confounds_tag : :obj:`str`, Defaults="desc-confounds_timeseries"
        Filename "suffix":
        If generating confounds, what path should they have?
        Defaults to `desc-confounds_timeseries` as in :term:`fMRIPrep` >= 20.2
        but can be other values (e.g. "desc-confounds_regressors" as
        in :term:`fMRIPrep` < 20.2).

    random_state : :obj:`int` or :obj:`numpy.random.RandomState` instance. \
                    Defaults=0
        Random number generator, or seed.

    entities : :obj:`dict`, optional
        Extra entity to add to the BIDS filename with a list of values.
        For example, if you want to add an 'echo' entity
        with values '1' for some files and '1' for others,
        you would pass: ``entities={"echo": ['1', '2']}``.

    Returns
    -------
    dataset directory name : :obj:`pathlib.Path`
        'bids_dataset'.

    Notes
    -----
        Creates a directory with dummy files.

    """
    n_voxels = 4

    rand_gen = check_random_state(random_state)

    bids_dataset_dir = "bids_dataset"
    bids_path = Path(base_dir) / bids_dataset_dir

    for task_ in tasks:
        _check_bids_label(task_)

    if (not isinstance(n_runs, list) 
        or not all(isinstance(x, int) for x in n_runs)):
        raise TypeError(
            "n_runs must be a list of integers."
        )

    if len(tasks) != len(n_runs):
        raise ValueError(
            "The number of tasks and number of runs must be the same."
            f"Got {len(tasks)} tasks and {len(n_runs)} runs."
        )

    if entities is None:
        entities = {}
    _check_entities_and_labels(entities)

    _mock_bids_dataset(
        bids_path=bids_path,
        n_sub=n_sub,
        n_ses=n_ses,
        tasks=tasks,
        n_runs=n_runs,
        entities=entities,
        n_voxels=n_voxels,
        rand_gen=rand_gen,
    )

    if with_derivatives:
        if not with_confounds:
            confounds_tag = None

        _mock_bids_derivatives(
            bids_path=bids_path,
            n_sub=n_sub,
            n_ses=n_ses,
            tasks=tasks,
            n_runs=n_runs,
            confounds_tag=confounds_tag,
            entities=entities,
            n_voxels=n_voxels,
            rand_gen=rand_gen,
        )

    return bids_path


def _check_entities_and_labels(entities):
    """Check entities and labels are BIDS compliant.
    
    Parameters
    ----------
    entities : :obj:`dict`, optional
        Extra entity to add to the BIDS filename with a list of values.
        For example, if you want to add an 'echo' entity
        with values '1' for some files and '1' for others,
        you would pass: ``entities={"echo": ['1', '2']}``.
    """
    if len(entities.keys()) > 1:
        # Generating dataset with more than one extra entity
        # becomes too complex.
        # Won't be implemented until there is a need.
        raise ValueError("Only a single extra entity is supported for now.")

    for key in entities:
        if key not in [*_bids_entities()["raw"], *_bids_entities()["derivatives"]]:
            raise ValueError(
                f"Invalid entity: {key}. Allowed entities are: "
                f"{[*_bids_entities()['raw'], *_bids_entities()['derivatives']]}"
            )
        [_check_bids_label(label_) for label_ in entities[key]]


def _mock_bids_dataset(
    bids_path,
    n_sub,
    n_ses,
    tasks,
    n_runs,
    entities,
    n_voxels,
    rand_gen,
):
    """Create a fake raw :term:`bids<BIDS>` dataset directory with dummy files.

    Parameters
    ----------
    base_dir : :obj:`Path`
        Path where to create the fake :term:`BIDS` dataset.

    n_sub : :obj:`int`
        Number of subjects to be simulated in the dataset.

    n_ses : :obj:`int`
        Number of sessions to be simulated in the dataset.
        Ignored if n_ses=0.

    tasks : :obj:`list` of :obj:`str`
        List of tasks to be simulated in the dataset.

    n_runs : :obj:`list` of :obj:`int`
        Number of runs to create, where each element indicates the
        number of runs for the corresponding task.
        No run entity will be used if a value is equal to 1 or less.

    entities : :obj:`dict`, optional
        Extra entities to add to the BIDS filename with a list of values.

    n_voxels : :obj:`int`
        Number of voxels along a given axis in the functional image.

    rand_gen : :obj:`numpy.random.RandomState` instance
        Random number generator.

    """
    bids_path.mkdir(parents=True, exist_ok=True)

    bids_path.joinpath("README.txt").write_text("")

    for subject, session in itertools.product(
        _listify(n_sub), _listify(n_ses)
    ):
        subses_dir = bids_path / f"sub-{subject}"
        if session != "":
            subses_dir = subses_dir / f"ses-{session}"

        if session in ("01", ""):
            _write_bids_raw_anat(subses_dir, subject, session)

        func_path = subses_dir / "func"
        func_path.mkdir(parents=True, exist_ok=True)

        for task, n_run in zip(tasks, n_runs):
            for run in _listify(n_run):

                if entities:
                    for key in entities:
                        for label in entities[key]:
                            fields = _init_fields(
                                subject=subject,
                                session=session,
                                task=task,
                                run=run,
                            )
                            if key in _bids_entities()["raw"]:
                                fields["entities"][key] = label
                            _write_bids_raw_func(
                                func_path=func_path,
                                fields=fields,
                                n_voxels=n_voxels,
                                rand_gen=rand_gen,
                            )

                else:
                    fields = _init_fields(
                        subject=subject, session=session, task=task, run=run
                    )
                    _write_bids_raw_func(
                        func_path=func_path,
                        fields=fields,
                        n_voxels=n_voxels,
                        rand_gen=rand_gen,
                    )


def _mock_bids_derivatives(
    bids_path,
    n_sub,
    n_ses,
    tasks,
    n_runs,
    confounds_tag,
    entities,
    n_voxels,
    rand_gen,
):
    """Create a fake raw :term:`bids<BIDS>` dataset directory with dummy files.

    Parameters
    ----------
    base_dir : :obj:`Path`
        Path where to create the fake :term:`BIDS` dataset.

    n_sub : :obj:`int`
        Number of subjects to be simulated in the dataset.

    n_ses : :obj:`int`
        Number of sessions to be simulated in the dataset.
        Ignored if n_ses=0.

    tasks : :obj:`list` of :obj:`str`
        List of tasks to be simulated in the dataset.

    n_runs : :obj:`list` of :obj:`int`
        Number of runs to create, where each element indicates the
        number of runs for the corresponding task.
        No run entity will be used if a value is equal to 1 or less.

    confounds_tag : :obj:`str`
        Filename "suffix":
        For example: `desc-confounds_timeseries`
        or "desc-confounds_regressors".

    entities : :obj:`dict`
        Extra entity to add to the BIDS filename with a list of values.

    n_voxels : :obj:`int`
        Number of voxels along a given axis in the functional image.

    rand_gen : :obj:`numpy.random.RandomState` instance
        Random number generator.

    """
    bids_path = bids_path / "derivatives"
    bids_path.mkdir(parents=True, exist_ok=True)

    for subject, session in itertools.product(
        _listify(n_sub), _listify(n_ses)
    ):
        subses_dir = bids_path / f"sub-{subject}"
        if session != "":
            subses_dir = subses_dir / f"ses-{session}"

        func_path = subses_dir / "func"
        func_path.mkdir(parents=True, exist_ok=True)

        for task, n_run in zip(tasks, n_runs):
            for run in _listify(n_run):

                if entities:
                    for key in entities:
                        for label in entities[key]:
                            fields = _init_fields(
                                subject=subject,
                                session=session,
                                task=task,
                                run=run,
                            )
                            fields["entities"][key] = label
                            _write_bids_derivative_func(
                                func_path=func_path,
                                fields=fields,
                                n_voxels=n_voxels,
                                rand_gen=rand_gen,
                                confounds_tag=confounds_tag,
                            )

                else:
                    fields = _init_fields(
                        subject=subject, session=session, task=task, run=run
                    )
                    _write_bids_derivative_func(
                        func_path=func_path,
                        fields=fields,
                        n_voxels=n_voxels,
                        rand_gen=rand_gen,
                        confounds_tag=confounds_tag,
                    )


def _listify(n):
    """Return a list of zero padded BIDS labels.

    If n is 0 or less, return an empty list.

    Parameters
    ----------
    n : :obj:`int`
        Number of labels to create.

    Returns
    -------
    List of labels : :obj:`list` of :obj:`str`

    """
    return [""] if n <= 0 else [f"{label:02}" for label in range(1, n + 1)]


def _create_bids_filename(
    fields, entities_to_include = None
):
    """Create BIDS filename from dictionary of entity-label pairs.

    Parameters
    ----------
    fields : :obj:`dict` of :obj:`str`
        Dictionary of entity-label pairs, for example:

        {
         "suffix": "T1w",
         "extension": "nii.gz",
         "entities": {"acq":  "ap",
                      "desc": "preproc"}
        }.

    Returns
    -------
    BIDS filename : :obj:`str`

    """
    if entities_to_include is None:
        entities_to_include = _bids_entities()["raw"]

    filename = ""

    for key in entities_to_include:
        if key in fields["entities"]:
            value = fields["entities"][key]
            if value not in (None, ""):
                filename += f"{key}-{value}_"
    filename += f"{fields['suffix']}.{fields['extension']}"

    return filename


def _init_fields(subject,
                 session,
                 task,
                 run):
    """Initialize fields to help create a valid BIDS filename.

    Parameters
    ----------
    subject : :obj:`str`
        Subject label

    session : :obj:`str`
        Session label

    task : :obj:`str`
        Task label

    run : :obj:`str`
        Run label

    Returns
    -------
    dict
        Fields used to create a BIDS filename.

    See Also
    --------
    _create_bids_filename

    """
    fields = {
        "suffix": "bold",
        "extension": "nii.gz",
        "entities": {
            "sub": subject,
            "ses": session,
            "task": task,
            "run": run,
        },
    }
    return fields


def _write_bids_raw_anat(subses_dir, subject, session) -> None:
    """Create a dummy anat T1w file.

    Parameters
    ----------
    subses_dir : :obj:`Path`
        Subject session directory

    subject : :obj:`str`
        Subject label

    session : :obj:`str`
        Session label
    """
    anat_path = subses_dir / "anat"
    anat_path.mkdir(parents=True, exist_ok=True)
    fields = {
        "suffix": "T1w",
        "extension": "nii.gz",
        "entities": {"sub": subject, "ses": session},
    }
    (anat_path / _create_bids_filename(fields)).write_text("")


def _write_bids_raw_func(
    func_path,
    fields,
    n_voxels,
    rand_gen,
):
    """Create BIDS functional raw nifti, json sidecar and events files.

    Parameters
    ----------
    func_path : :obj:`Path`
        Path to a subject functional directory.

    file_id : :obj:`str`
        Root of the BIDS filename:
        typically basename without the BIDS suffix and extension.

    n_voxels : :obj:`int`
        Number of voxels along a given axis in the functional image.

    rand_gen : :obj:`numpy.random.RandomState` instance
        Random number generator.

    """
    n_time_points = 100
    bold_path = func_path / _create_bids_filename(fields)
    write_fake_bold_img(
        bold_path,
        [n_voxels, n_voxels, n_voxels, n_time_points],
        random_state=rand_gen,
    )

    repetition_time = 1.5
    fields["extension"] = "json"
    param_path = func_path / _create_bids_filename(fields)
    param_path.write_text(json.dumps({"RepetitionTime": repetition_time}))

    fields["suffix"] = "events"
    fields["extension"] = "tsv"
    events_path = func_path / _create_bids_filename(fields)
    basic_paradigm().to_csv(events_path, sep="\t", index=None)


def _write_bids_derivative_func(
    func_path,
    fields,
    n_voxels,
    rand_gen,
    confounds_tag,
):
    """Create BIDS functional derivative and confounds files.

    Files created come with two spaces and descriptions.
    Spaces are: 'MNI' and 'T1w'.
    Descriptions are: 'preproc' and :term:`fMRIPrep`.
    Only space 'T1w' include both descriptions.

    Parameters
    ----------
    func_path : :obj:`Path`
        Path to a subject functional directory.

    file_id : :obj:`str`
        Root of the BIDS filename:
        typically basename without the BIDS suffix and extension.

    n_voxels : :obj:`int`
        Number of voxels along a given axis in the functional image.

    rand_gen : :obj:`numpy.random.RandomState` instance
        Random number generator.

    confounds_tag : :obj:`str`, optional.
        Filename "suffix":
        For example: `desc-confounds_timeseries`
        or "desc-confounds_regressors".

    """
    n_time_points = 100

    if confounds_tag is not None:
        fields["suffix"] = confounds_tag
        fields["extension"] = "tsv"
        confounds_path = func_path / _create_bids_filename(
            fields=fields, entities_to_include=_bids_entities()["raw"]
        )
        basic_confounds(length=n_time_points, random_state=rand_gen).to_csv(
            confounds_path, sep="\t", index=None
        )

    fields["suffix"] = "bold"
    fields["extension"] = "nii.gz"

    shape = [n_voxels, n_voxels, n_voxels, n_time_points]

    for space in ("MNI", "T1w"):
        for desc in ("preproc", "fmriprep"):
            # Only space 'T1w' include both descriptions.
            if space == "MNI" and desc == "fmriprep":
                continue

            fields["entities"]["space"] = space
            fields["entities"]["desc"] = desc

            entities_to_include = [
                *_bids_entities()["raw"], *_bids_entities()["derivatives"]
            ]

            bold_path = func_path / _create_bids_filename(
                fields=fields, entities_to_include=entities_to_include
            )
            write_fake_bold_img(bold_path, shape=shape, random_state=rand_gen)
