"""
Data generation utilities
"""
import json
import os
import string

import numpy as np
import pandas as pd
import scipy.signal
from scipy import ndimage

from sklearn.utils import check_random_state
import scipy.linalg
import nibabel

from nibabel import Nifti1Image

from .. import masking
from . import logger
from nilearn import datasets, image, input_data


def generate_mni_space_img(n_scans=1, res=30, random_state=0, mask_dilation=2):
    rng = check_random_state(random_state)
    mni = datasets.load_mni152_brain_mask()
    target_affine = np.eye(3) * res
    mask_img = image.resample_img(
        mni, target_affine=target_affine, interpolation="nearest")
    masker = input_data.NiftiMasker(mask_img).fit()
    n_voxels = image.get_data(mask_img).sum()
    data = rng.randn(n_scans, n_voxels)
    if mask_dilation is not None and mask_dilation > 0:
        mask_img = image.new_img_like(
            mask_img, ndimage.binary_dilation(
                image.get_data(mask_img), iterations=mask_dilation))
    return masker.inverse_transform(data), mask_img


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
    n_features : int
        Number of features.

    n_regions : int
        Number of regions.

    overlap : int, optional
        Number of overlapping voxels between two regions (more or less).
        Default=0.

    window : str, optional
        Name of a window in scipy.signal. e.g. "hamming".
        Default='boxcar'.

    Returns
    -------
    regions : numpy.ndarray
        Regions, nepresented as signals.
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
    n_regions : int
        Number of regions to generate.

    overlap : int, optional
        Approximate number of voxels common to two neighboring regions.
        Default=0.

    window : str, optional
        Name of a window in scipy.signal. Used to get non-uniform regions.
        Default='boxcar'.

    ran_gen : None or numpy.random.RandomState, optional
        Random generator to use for generation. Default=None.

    affine : Numpy array, optional
        Affine transformation to use. Default=np.eye(4).

    border : int, optional
        Number of background voxels on each side of the 3D volumes.
        Default=1.

    Returns
    -------
    maps : nibabel.Nifti1Image
        4D array, containing maps.

    mask_img : nibabel.Nifti1Image
        3D, The mask outside of which maps are 0.

    """
    mask = np.zeros(shape, dtype=np.int8)
    mask[border:-border, border:-border, border:-border] = 1
    ts = generate_regions_ts(mask.sum(), n_regions, overlap=overlap,
                             rand_gen=rand_gen, window=window)
    mask_img = nibabel.Nifti1Image(mask, affine)
    return masking.unmask(ts, mask_img), mask_img


def generate_labeled_regions(shape, n_regions, rand_gen=None, labels=None,
                             affine=np.eye(4), dtype=int):
    """Generate a 3D volume with labeled regions.

    Parameters
    ----------
    shape : tuple
        Shape of returned array.

    n_regions : int
        Number of regions to generate. By default (if "labels" is None),
        add a background with value zero.

    labels : iterable, optional
        Labels to use for each zone. If provided, n_regions is unused.

    rand_gen : numpy.random.RandomState
        Random generator to use for generation.

    affine : numpy.ndarray, optional
        Affine of returned image. Default=np.eye(4).

    dtype : type, optional
        Data type of image. Default=int.

    Returns
    -------
    regions : nibabel.Nifti1Image
        Data has shape "shape", containing region labels.

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
    data[np.ones(shape, dtype=bool)] = regions.sum(axis=0).T
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
    shape : tuple, optional
        Shape of 3D volume. Default=(10, 11, 12).

    length : int, optional
        Number of time instants. Default=17.

    kind : string, optional
        Kind of signal used as timeseries.
        "noise": uniformly sampled values in [0..255]
        "step": 0.5 for the first half then 1.
        Default='noise'.

    affine : numpy.ndarray, optional
        Affine of returned images. Default=np.eye(4).

    n_blocks : int or None, optional
        Number of condition blocks.

    block_size : int or None, optional
        Number of timepoints in a block. Used only if n_blocks is not
        None. Defaults to 3 if n_blocks is not None.

    block_type : str, optional
        Defines if the returned target should be used for
        'classification' or 'regression'.
        Default='classification'.

    rand_gen : numpy.random.RandomState, optional
        Random generator to use for generation.
        Default=np.random.RandomState(0).

    Returns
    -------
    fmri : nibabel.Nifti1Image
        Fake fmri signal.
        shape: shape + (length,)

    mask : nibabel.Nifti1Image
        Mask giving non-zero voxels.

    target : numpy.ndarray
        Classification or regression target. Shape of number of
        time points (length). Returned only if n_blocks is not None.

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
    flat_fmri = fmri[mask.astype(bool)]
    flat_fmri /= np.abs(flat_fmri).max()
    target = np.zeros(length, dtype=int)
    rest_max_size = (length - (n_blocks * block_size)) // n_blocks
    if rest_max_size < 0:
        raise ValueError(
            '%s is too small '
            'to put %s blocks of size %s' % (
                length, n_blocks, block_size))
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
            trials_effect = (
                rand_gen.random_sample(block_size) + 1) * block
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
    return (nibabel.Nifti1Image(fmri, affine),
            nibabel.Nifti1Image(mask, affine), target)


def generate_fake_fmri_data_and_design(shapes, rk=3, affine=np.eye(4)):
    fmri_data = []
    design_matrices = []
    for i, shape in enumerate(shapes):
        data = np.random.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        fmri_data.append(Nifti1Image(data, affine))
        columns = np.random.choice(list(string.ascii_lowercase),
                                   size=rk,
                                   replace=False)
        design_matrices.append(pd.DataFrame(np.random.randn(shape[3], rk),
                                            columns=columns))
    mask = Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                       affine)
    return mask, fmri_data, design_matrices


def write_fake_fmri_data_and_design(shapes, rk=3, affine=np.eye(4)):
    mask_file, fmri_files, design_files = 'mask.nii', [], []
    for i, shape in enumerate(shapes):
        fmri_files.append('fmri_run%d.nii' % i)
        data = np.random.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        Nifti1Image(data, affine).to_filename(fmri_files[-1])
        design_files.append('dmtx_%d.csv' % i)
        pd.DataFrame(np.random.randn(shape[3], rk),
                     columns=['', '', '']).to_csv(design_files[-1])
    Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                affine).to_filename(mask_file)
    return mask_file, fmri_files, design_files


def write_fake_bold_img(file_path, shape, rk=3, affine=np.eye(4)):
    data = np.random.randn(*shape)
    data[1:-1, 1:-1, 1:-1] += 100
    Nifti1Image(data, affine).to_filename(file_path)
    return file_path


def generate_signals_from_precisions(precisions,
                                     min_n_samples=50, max_n_samples=100,
                                     random_state=0):
    """Generate timeseries according to some given precision matrices.

    Signals all have zero mean.

    Parameters
    ----------
    precisions : list of numpy.ndarray
        List of precision matrices. Every matrix must be square (with the same
        size) and positive definite. The output of
        generate_group_sparse_gaussian_graphs() can be used here.

    min_samples, max_samples : int, optional
        The number of samples drawn for each timeseries is taken at random
        between these two numbers. Defaults are 50 and 100.

    random_state : int, optional
        Random state. Default=0.

    Returns
    -------
    signals : list of numpy.ndarray
        Output signals. signals[n] corresponds to precisions[n], and has shape
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
        Number of subjects. Default=5.

    n_features : int, optional
        Number of signals per subject to generate. Default=30.

    density : float, optional
        Density of edges in graph topology. Default=0.1.

    min_n_samples, max_n_samples : int, optional
        Each subject have a different number of samples, between these two
        numbers. All signals for a given subject have the same number of
        samples. Defaults are 30 and 50.

    random_state : int or numpy.random.RandomState instance, optional
        Random number generator, or seed. Default=0.

    verbose : int, optional
        Verbosity level (0 means no message). Default=0.

    Returns
    -------
    subjects : list of numpy.ndarray, shape for each (n_samples, n_features)
        subjects[n] is the signals for subject n. They are provided as a numpy
        len(subjects) = n_subjects. n_samples varies according to the subject.

    precisions : list of numpy.ndarray
        Precision matrices.

    topology : numpy.ndarray
        Binary array giving the graph topology used for generating covariances
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


def basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    events = pd.DataFrame({'trial_type': conditions,
                           'onset': onsets})
    return events


def basic_confounds(length):
    columns = ['RotX', 'RotY', 'RotZ', 'X', 'Y', 'Z']
    data = np.random.rand(length, 6)
    confounds = pd.DataFrame(data, columns=columns)
    return confounds


def create_fake_bids_dataset(base_dir='', n_sub=10, n_ses=2,
                             tasks=['localizer', 'main'],
                             n_runs=[1, 3], with_derivatives=True,
                             with_confounds=True,
                             confounds_tag="desc-confounds_timeseries",
                             no_session=False):
    """Creates a fake :term:`bids<BIDS>` dataset directory with dummy files.
    Returns fake dataset directory name.

    Parameters
    ----------
    base_dir : string (Absolute path), optional
        Absolute directory path in which to create the fake :term:`BIDS` dataset dir.
        Default: Current directory.

    n_sub : int, optional
        Number of subject to be simulated in the dataset.
        Default=10.

    n_ses : int, optional
        Number of sessions to be simulated in the dataset.
        Ignored if no_session=True.
        Default=2.

    tasks : List of str, optional
        List of tasks. Default=["localizer", "main"].

    n_runs : List of int, optional
        Default=[1, 3].

    with_derivatives : bool, optional
        In the case derivatives are included, they come with two spaces and
        descriptions. Spaces are 'MNI' and 'T1w'. Descriptions are 'preproc'
        and 'fmriprep'. Only space 'T1w' include both descriptions.
        Default=True.

    with_confounds : bool, optional
        Default=True.

    confounds_tag : string (filename suffix), optional
        If generating confounds, what path should they have? Defaults to
        `desc-confounds_timeseries` as in `fmriprep` >= 20.2 but can be other
        values (e.g. "desc-confounds_regressors" as in `fmriprep` < 20.2)
        Default="desc-confounds_timeseries".

    no_session : bool, optional
        Specifying no_sessions will only produce runs and files without the
        optional session field. In this case n_ses will be ignored.
        Default=False.

    Returns
    -------
    dataset directory name : string
        'bids_dataset'.

    Notes
    -----
        Creates a directory with dummy files.

    """
    bids_path = os.path.join(base_dir, 'bids_dataset')
    os.makedirs(bids_path)
    # Create surface bids dataset
    open(os.path.join(bids_path, 'README.txt'), 'w')
    vox = 4
    created_sessions = ['ses-%02d' % label for label in range(1, n_ses + 1)]
    if no_session:
        created_sessions = ['']
    for subject in ['sub-%02d' % label for label in range(1, n_sub + 1)]:
        for session in created_sessions:
            subses_dir = os.path.join(bids_path, subject, session)
            if session == 'ses-01' or session == '':
                anat_path = os.path.join(subses_dir, 'anat')
                os.makedirs(anat_path)
                anat_file = os.path.join(anat_path, subject + '_T1w.nii.gz')
                open(anat_file, 'w')
            func_path = os.path.join(subses_dir, 'func')
            os.makedirs(func_path)
            for task, n_run in zip(tasks, n_runs):
                run_labels = [
                    'run-%02d' % label for label in range(1, n_run + 1)]
                for run in run_labels:
                    fields = [subject, session, 'task-' + task]
                    if '' in fields:
                        fields.remove('')
                    file_id = '_'.join(fields)
                    if n_run > 1:
                        file_id += '_' + run
                    bold_path = os.path.join(func_path,
                                             file_id + '_bold.nii.gz')
                    write_fake_bold_img(bold_path, [vox, vox, vox, 100])
                    events_path = os.path.join(func_path,
                                               file_id + '_events.tsv')
                    basic_paradigm().to_csv(events_path,
                                            sep='\t',
                                            index=None)
                    param_path = os.path.join(func_path,
                                              file_id + '_bold.json')
                    with open(param_path, 'w') as param_file:
                        json.dump({'RepetitionTime': 1.5}, param_file)

    # Create derivatives files
    if with_derivatives:
        bids_path = os.path.join(base_dir, 'bids_dataset', 'derivatives')
        os.makedirs(bids_path)
        for subject in ['sub-%02d' % label for label in range(1, 11)]:
            for session in created_sessions:
                subses_dir = os.path.join(bids_path, subject, session)
                func_path = os.path.join(subses_dir, 'func')
                os.makedirs(func_path)
                for task, n_run in zip(tasks, n_runs):
                    for run in ['run-%02d' % label
                                for label in range(1, n_run + 1)
                                ]:
                        fields = [subject, session, 'task-' + task]
                        if '' in fields:
                            fields.remove('')
                        file_id = '_'.join(fields)
                        if n_run > 1:
                            file_id += '_' + run
                        preproc = (
                            file_id + '_space-MNI_desc-preproc_bold.nii.gz'
                        )
                        preproc_path = os.path.join(func_path, preproc)
                        write_fake_bold_img(preproc_path,
                                            [vox, vox, vox, 100]
                                            )
                        preproc = (
                            file_id + '_space-T1w_desc-preproc_bold.nii.gz'
                        )
                        preproc_path = os.path.join(func_path, preproc)
                        write_fake_bold_img(preproc_path,
                                            [vox, vox, vox, 100]
                                            )
                        preproc = (
                            file_id + '_space-T1w_desc-fmriprep_bold.nii.gz'
                        )
                        preproc_path = os.path.join(func_path, preproc)
                        write_fake_bold_img(preproc_path,
                                            [vox, vox, vox, 100]
                                            )
                        if with_confounds:
                            confounds_path = os.path.join(
                                func_path,
                                file_id + '_' + confounds_tag + '.tsv',
                            )
                            basic_confounds(100).to_csv(confounds_path,
                                                        sep='\t', index=None)
    return 'bids_dataset'
