"""
This module presents an interface to use the glm implemented in
nistats.regression.

It contains the GLM and contrast classes that are meant to be the main objects
of fMRI data analyses.

Author: Bertrand Thirion, Martin Perez-Guevara, 2016

"""
import glob
import json
import os
import sys
import time
from warnings import warn

import numpy as np
import pandas as pd
from nibabel import Nifti1Image

from sklearn.base import (BaseEstimator,
                          clone,
                          TransformerMixin,
                          )
from sklearn.externals.joblib import Memory
from nilearn.input_data import NiftiMasker
from nilearn._utils import CacheMixin
from nilearn._utils.niimg_conversions import check_niimg
from sklearn.externals.joblib import (Parallel,
                                      delayed,
                                      )

from .contrasts import _compute_fixed_effect_contrast, expression_to_contrast_vector
from .design_matrix import make_first_level_design_matrix
from .regression import (ARModel,
                         OLSModel,
                         SimpleRegressionResults,
                         )
from .utils import (_basestring,
                    _check_run_tables,
                    _check_events_file_uses_tab_separators,
                    get_bids_files,
                    parse_bids_filename,
                    get_data
                    )
from nistats._utils.helpers import replace_parameters


def mean_scaling(Y, axis=0):
    """Scaling of the data to have percent of baseline change along the
    specified axis

    Parameters
    ----------
    Y : array of shape (n_time_points, n_voxels)
       The input data.

    Returns
    -------
    Y : array of shape (n_time_points, n_voxels),
       The data after mean-scaling, de-meaning and multiplication by 100.

    mean : array of shape (n_voxels,)
        The data mean.

    """
    mean = Y.mean(axis=axis)
    if (mean == 0).any():
        warn('Mean values of 0 observed.'
             'The data have probably been centered.'
             'Scaling might not work as expected')
    mean = np.maximum(mean, 1)
    Y = 100 * (Y / mean - 1)
    return Y, mean


def _ar_model_fit(X, val, Y):
    """Wrapper for fit method of ARModel to allow joblib parallelization"""
    return ARModel(X, val).fit(Y)


def run_glm(Y, X, noise_model='ar1', bins=100, n_jobs=1, verbose=0):
    """ GLM fit for an fMRI data matrix

    Parameters
    ----------
    Y : array of shape (n_time_points, n_voxels)
        The fMRI data.

    X : array of shape (n_time_points, n_regressors)
        The design matrix.

    noise_model : {'ar1', 'ols'}, optional
        The temporal variance model. Defaults to 'ar1'.

    bins : int, optional
        Maximum number of discrete bins for the AR(1) coef histogram.

    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : int, optional
        The verbosity level. Defaut is 0

    Returns
    -------
    labels : array of shape (n_voxels,),
        A map of values on voxels used to identify the corresponding model.

    results : dict,
        Keys correspond to the different labels values
        values are RegressionResults instances corresponding to the voxels.

    """
    acceptable_noise_models = ['ar1', 'ols']
    if noise_model not in acceptable_noise_models:
        raise ValueError(
            "Acceptable noise models are {0}. You provided 'noise_model={1}'".\
                format(acceptable_noise_models, noise_model))

    if Y.shape[0] != X.shape[0]:
        raise ValueError(
            'The number of rows of Y should match the number of rows of X.'
            ' You provided X with shape {0} and Y with shape {1}'.\
                format(X.shape, Y.shape))

    # Create the model
    ols_result = OLSModel(X).fit(Y)

    if noise_model == 'ar1':
        # compute and discretize the AR1 coefs
        ar1 = ((ols_result.resid[1:] * ols_result.resid[:-1]).sum(axis=0) /
               (ols_result.resid ** 2).sum(axis=0))
        del ols_result
        ar1 = (ar1 * bins).astype(np.int) * 1. / bins
        # Fit the AR model acccording to current AR(1) estimates
        results = {}
        labels = ar1
        # Parallelize by creating a job per ARModel
        vals = np.unique(ar1)
        ar_result = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_ar_model_fit)(X, val, Y[:, labels == val]) for val in vals)
        for val, result in zip(vals, ar_result):
            results[val] = result
        del vals
        del ar_result

    else:
        labels = np.zeros(Y.shape[1])
        results = {0.0: ols_result}

    return labels, results


class FirstLevelModel(BaseEstimator, TransformerMixin, CacheMixin):
    """ Implementation of the General Linear Model for single session fMRI data

    Parameters
    ----------
    t_r : float
        This parameter indicates repetition times of the experimental runs.
        In seconds. It is necessary to correctly consider times in the design
        matrix. This parameter is also passed to nilearn.signal.clean.
        Please see the related documentation for details.

    slice_time_ref : float, optional (default 0.)
        This parameter indicates the time of the reference slice used in the
        slice timing preprocessing step of the experimental runs. It is
        expressed as a percentage of the t_r (time repetition), so it can have
        values between 0. and 1.

    hrf_model : {'spm', 'spm + derivative', 'spm + derivative + dispersion',
        'glover', 'glover + derivative', 'glover + derivative + dispersion',
        'fir', None}
        String that specifies the hemodynamic response function. Defaults to 'glover'.

    drift_model : string, optional
        This parameter specifies the desired drift model for the design
        matrices. It can be 'polynomial', 'cosine' or None.

    high_pass : float, optional
        This parameter specifies the cut frequency of the high-pass filter in
        Hz for the design matrices. Used only if drift_model is 'cosine'.

    drift_order : int, optional
        This parameter specifices the order of the drift model (in case it is
        polynomial) for the design matrices.

    fir_delays : array of shape(n_onsets) or list, optional
        In case of FIR design, yields the array of delays used in the FIR
        model, in scans.

    min_onset : float, optional
        This parameter specifies the minimal onset relative to the design
        (in seconds). Events that start before (slice_time_ref * t_r +
        min_onset) are not considered.

    mask_img : Niimg-like, NiftiMasker object or False, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a NiftiMasker with default
        parameters. If False is given then the data will not be masked.

    target_affine : 3x3 or 4x4 matrix, optional
        This parameter is passed to nilearn.image.resample_img. Please see the
        related documentation for details.

    target_shape : 3-tuple of integers, optional
        This parameter is passed to nilearn.image.resample_img. Please see the
        related documentation for details.

    smoothing_fwhm : float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    memory : string, optional
        Path to the directory used to cache the masking process and the glm
        fit. By default, no caching is done. Creates instance of joblib.Memory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    signal_scaling : False, int or (int, int), optional,
        If not False, fMRI signals are scaled to the mean value of scaling_axis
        given, which can be 0, 1 or (0, 1). 0 refers to mean scaling each voxel
        with respect to time, 1 refers to mean scaling each time point with
        respect to all voxels and (0, 1) refers to scaling with respect to
        voxels and time, which is known as grand mean scaling.
        Incompatible with standardize (standardize=False is enforced when
        signal_scaling is not False).

    noise_model : {'ar1', 'ols'}, optional
        The temporal variance model. Defaults to 'ar1'

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        If 0 prints nothing. If 1 prints progress by computation of
        each run. If 2 prints timing details of masker and GLM. If 3
        prints masker computation details.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    minimize_memory : boolean, optional
        Gets rid of some variables on the model fit results that are not
        necessary for contrast computation and would only be useful for
        further inspection of model details. This has an important impact
        on memory consumption. True by default.

    subject_label : string, optional
        This id will be used to identify a `FirstLevelModel` when passed to
        a `SecondLevelModel` object.

    Attributes
    ----------
    labels_ : array of shape (n_voxels,),
        a map of values on voxels used to identify the corresponding model

    results_ : dict,
        with keys corresponding to the different labels values.
        Values are SimpleRegressionResults corresponding to the voxels,
        if minimize_memory is True,
        RegressionResults if minimize_memory is False


    """
    @replace_parameters({'mask': 'mask_img'}, end_version='next')
    def __init__(self, t_r=None, slice_time_ref=0., hrf_model='glover',
                 drift_model='cosine', high_pass=.01, drift_order=1,
                 fir_delays=[0], min_onset=-24, mask_img=None, target_affine=None,
                 target_shape=None, smoothing_fwhm=None, memory=Memory(None),
                 memory_level=1, standardize=False, signal_scaling=0,
                 noise_model='ar1', verbose=0, n_jobs=1,
                 minimize_memory=True, subject_label=None):
        # design matrix parameters
        self.t_r = t_r
        self.slice_time_ref = slice_time_ref
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.high_pass = high_pass
        self.drift_order = drift_order
        self.fir_delays = fir_delays
        self.min_onset = min_onset
        # glm parameters
        self.mask_img = mask_img
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.smoothing_fwhm = smoothing_fwhm
        if isinstance(memory, _basestring):
            self.memory = Memory(memory)
        else:
            self.memory = memory
        self.memory_level = memory_level
        self.standardize = standardize
        if signal_scaling is False:
            self.signal_scaling = signal_scaling
        elif signal_scaling in [0, 1, (0, 1)]:
            self.scaling_axis = signal_scaling
            self.signal_scaling = True
            self.standardize = False
        else:
            raise ValueError('signal_scaling must be "False", "0", "1"'
                             ' or "(0, 1)"')
        self.noise_model = noise_model
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.minimize_memory = minimize_memory
        # attributes
        self.labels_ = None
        self.results_ = None
        self.subject_label = subject_label

    def fit(self, run_imgs, events=None, confounds=None,
            design_matrices=None):
        """ Fit the GLM

        For each run:
        1. create design matrix X
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        run_imgs: Niimg-like object or list of Niimg-like objects,
            See http://nilearn.github.io/manipulating_images/input_output.html#inputing-data-file-names-or-image-objects
            Data on which the GLM will be fitted. If this is a list,
            the affine is considered the same for all.

        events: pandas Dataframe or string or list of pandas DataFrames or
                   strings

            fMRI events used to build design matrices. One events object
            expected per run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        confounds: pandas Dataframe or string or list of pandas DataFrames or
                   strings

            Each column in a DataFrame corresponds to a confound variable
            to be included in the regression model of the respective run_img.
            The number of rows must match the number of volumes in the
            respective run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        design_matrices: pandas DataFrame or list of pandas DataFrames,
            Design matrices that will be used to fit the GLM. If given it
            takes precedence over events and confounds.

        """
        # Check arguments
        # Check imgs type
        if events is not None:
            _check_events_file_uses_tab_separators(
                events_files=events)
        if not isinstance(run_imgs, (list, tuple)):
            run_imgs = [run_imgs]
        if design_matrices is None:
            if events is None:
                raise ValueError('events or design matrices must be provided')
            if self.t_r is None:
                raise ValueError('t_r not given to FirstLevelModel object'
                                 ' to compute design from events')
        else:
            design_matrices = _check_run_tables(run_imgs, design_matrices,
                                                'design_matrices')
        # Check that number of events and confound files match number of runs
        # Also check that events and confound files can be loaded as DataFrame
        if events is not None:
            events = _check_run_tables(run_imgs, events, 'events')
        if confounds is not None:
            confounds = _check_run_tables(run_imgs, confounds, 'confounds')

        # Learn the mask
        if self.mask_img is False:
            # We create a dummy mask to preserve functionality of api
            ref_img = check_niimg(run_imgs[0])
            self.mask_img = Nifti1Image(np.ones(ref_img.shape[:3]),
                                        ref_img.affine)
        if not isinstance(self.mask_img, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask_img, smoothing_fwhm=self.smoothing_fwhm,
                target_affine=self.target_affine,
                standardize=self.standardize, mask_strategy='epi',
                t_r=self.t_r, memory=self.memory,
                verbose=max(0, self.verbose - 2),
                target_shape=self.target_shape,
                memory_level=self.memory_level)
            self.masker_.fit(run_imgs[0])
        else:
            if self.mask_img.mask_img_ is None and self.masker_ is None:
                self.masker_ = clone(self.mask_img)
                for param_name in ['target_affine', 'target_shape',
                                   'smoothing_fwhm', 't_r', 'memory',
                                   'memory_level']:
                    our_param = getattr(self, param_name)
                    if our_param is None:
                        continue
                    if getattr(self.masker_, param_name) is not None:
                        warn('Parameter %s of the masker'
                             ' overriden' % param_name)
                    setattr(self.masker_, param_name, our_param)
                self.masker_.fit(run_imgs[0])
            else:
                self.masker_ = self.mask_img

        # For each run fit the model and keep only the regression results.
        self.labels_, self.results_, self.design_matrices_ = [], [], []
        n_runs = len(run_imgs)
        t0 = time.time()
        for run_idx, run_img in enumerate(run_imgs):
            # Report progress
            if self.verbose > 0:
                percent = float(run_idx) / n_runs
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                if run_idx == 0:
                    remaining = 'go take a coffee, a big one'
                else:
                    remaining = (100. - percent) / max(0.01, percent) * dt
                    remaining = '%i seconds remaining' % remaining

                sys.stderr.write(
                    "Computing run %d out of %d runs (%s)\n"
                    % (run_idx + 1, n_runs, remaining))

            # Build the experimental design for the glm
            run_img = check_niimg(run_img, ensure_ndim=4)
            if design_matrices is None:
                n_scans = get_data(run_img).shape[3]
                if confounds is not None:
                    confounds_matrix = confounds[run_idx].values
                    if confounds_matrix.shape[0] != n_scans:
                        raise ValueError('Rows in confounds does not match'
                                         'n_scans in run_img at index %d'
                                         % (run_idx,))
                    confounds_names = confounds[run_idx].columns.tolist()
                else:
                    confounds_matrix = None
                    confounds_names = None
                start_time = self.slice_time_ref * self.t_r
                end_time = (n_scans - 1 + self.slice_time_ref) * self.t_r
                frame_times = np.linspace(start_time, end_time, n_scans)
                design = make_first_level_design_matrix(frame_times, events[run_idx],
                                                        self.hrf_model, self.drift_model,
                                                        self.high_pass, self.drift_order,
                                                        self.fir_delays, confounds_matrix,
                                                        confounds_names, self.min_onset)
            else:
                design = design_matrices[run_idx]
            self.design_matrices_.append(design)

            # Mask and prepare data for GLM
            if self.verbose > 1:
                t_masking = time.time()
                sys.stderr.write('Starting masker computation \r')

            Y = self.masker_.transform(run_img)
            del run_img  # Delete unmasked image to save memory

            if self.verbose > 1:
                t_masking = time.time() - t_masking
                sys.stderr.write('Masker took %d seconds       \n' % t_masking)

            if self.signal_scaling:
                Y, _ = mean_scaling(Y, self.scaling_axis)
            if self.memory:
                mem_glm = self.memory.cache(run_glm, ignore=['n_jobs'])
            else:
                mem_glm = run_glm

            # compute GLM
            if self.verbose > 1:
                t_glm = time.time()
                sys.stderr.write('Performing GLM computation\r')
            labels, results = mem_glm(Y, design.values,
                                      noise_model=self.noise_model,
                                      bins=100, n_jobs=self.n_jobs)
            if self.verbose > 1:
                t_glm = time.time() - t_glm
                sys.stderr.write('GLM took %d seconds         \n' % t_glm)

            self.labels_.append(labels)
            # We save memory if inspecting model details is not necessary
            if self.minimize_memory:
                for key in results:
                    results[key] = SimpleRegressionResults(results[key])
            self.results_.append(results)
            del Y

        # Report progress
        if self.verbose > 0:
            sys.stderr.write("\nComputation of %d runs done in %i seconds\n\n"
                             % (n_runs, time.time() - t0))

        return self

    def compute_contrast(self, contrast_def, stat_type=None,
                         output_type='z_score'):
        """Generate different outputs corresponding to
        the contrasts provided e.g. z_map, t_map, effects and variance.
        In multi-session case, outputs the fixed effects map.

        Parameters
        ----------
        contrast_def : str or array of shape (n_col) or list of (string or
                       array of shape (n_col))

            where ``n_col`` is the number of columns of the design matrix,
            (one array per run). If only one array is provided when there
            are several runs, it will be assumed that the same contrast is
            desired for all runs. The string can be a formula compatible with
            `pandas.DataFrame.eval`. Basically one can use the name of the
            conditions as they appear in the design matrix of the fitted model
            combined with operators +- and combined with numbers with operators
            +-`*`/.

        stat_type : {'t', 'F'}, optional
            type of the contrast

        output_type : str, optional
            Type of the output map. Can be 'z_score', 'stat', 'p_value',
            'effect_size', 'effect_variance' or 'all'

        Returns
        -------
        output : Nifti1Image or dict
            The desired output image(s). If ``output_type == 'all'``, then
            the output is a dictionary of images, keyed by the type of image.

        """
        if self.labels_ is None or self.results_ is None:
            raise ValueError('The model has not been fit yet')

        if isinstance(contrast_def, (np.ndarray, str)):
            con_vals = [contrast_def]
        elif isinstance(contrast_def, (list, tuple)):
            con_vals = contrast_def
        else:
            raise ValueError('contrast_def must be an array or str or list of'
                             ' (array or str)')

        # Translate formulas to vectors
        design_columns = self.design_matrices_[0].columns.tolist()
        for cidx, con in enumerate(con_vals):
            if isinstance(con, _basestring):
                con_vals[cidx] = expression_to_contrast_vector(
                    con, design_columns)

        n_runs = len(self.labels_)
        if len(con_vals) != n_runs:
            warn('One contrast given, assuming it for all %d runs' % n_runs)
            con_vals = con_vals * n_runs

        # 'all' is assumed to be the final entry; if adding more, place before 'all'
        valid_types = ['z_score', 'stat', 'p_value', 'effect_size',
                       'effect_variance', 'all']
        if output_type not in valid_types:
            raise ValueError('output_type must be one of {}'.format(valid_types))

        contrast = _compute_fixed_effect_contrast(self.labels_, self.results_,
                                                  con_vals, stat_type)

        output_types = valid_types[:-1] if output_type == 'all' else [output_type]

        outputs = {}
        for output_type_ in output_types:
            estimate_ = getattr(contrast, output_type_)()
            # Prepare the returned images
            output = self.masker_.inverse_transform(estimate_)
            contrast_name = str(con_vals)
            output.header['descrip'] = (
                '%s of contrast %s' % (output_type_, contrast_name))
            outputs[output_type_] = output

        return outputs if output_type == 'all' else output


@replace_parameters({'mask': 'mask_img'}, end_version='next')
def first_level_models_from_bids(
        dataset_path, task_label, space_label=None, img_filters=None,
        t_r=None, slice_time_ref=0., hrf_model='glover', drift_model='cosine',
        high_pass=.01, drift_order=1, fir_delays=[0], min_onset=-24,
        mask_img=None, target_affine=None, target_shape=None, smoothing_fwhm=None,
        memory=Memory(None), memory_level=1, standardize=False,
        signal_scaling=0, noise_model='ar1', verbose=0, n_jobs=1,
        minimize_memory=True, derivatives_folder='derivatives'):
    """Create FirstLevelModel objects and fit arguments from a BIDS dataset.

    It t_r is not specified this function will attempt to load it from a
    bold.json file alongside slice_time_ref. Otherwise t_r and slice_time_ref
    are taken as given.

    Parameters
    ----------
    dataset_path: str
        Directory of the highest level folder of the BIDS dataset. Should
        contain subject folders and a derivatives folder.

    task_label: str
        Task_label as specified in the file names like _task-<task_label>_.

    space_label: str, optional
        Specifies the space label of the preprocessed bold.nii images.
        As they are specified in the file names like _space-<space_label>_.

    img_filters: list of tuples (str, str), optional (default: None)
        Filters are of the form (field, label). Only one filter per field
        allowed. A file that does not match a filter will be discarded.
        Possible filters are 'acq', 'ce', 'dir', 'rec', 'run', 'echo', 'res',
        'den', and 'desc'. Filter examples would be ('desc', 'preproc'),
        ('dir', 'pa') and ('run', '10').

    derivatives_folder: str, optional
        derivatives and app folder path containing preprocessed files.
        Like "derivatives/FMRIPREP". default is simply "derivatives".

    All other parameters correspond to a `FirstLevelModel` object, which
    contains their documentation. The subject label of the model will be
    determined directly from the BIDS dataset.

    Returns
    -------
    models: list of `FirstLevelModel` objects
        Each FirstLevelModel object corresponds to a subject. All runs from
        different sessions are considered together for the same subject to run
        a fixed effects analysis on them.

    models_run_imgs: list of list of Niimg-like objects,
        Items for the FirstLevelModel fit function of their respective model.

    models_events: list of list of pandas DataFrames,
        Items for the FirstLevelModel fit function of their respective model.

    models_confounds: list of list of pandas DataFrames or None,
        Items for the FirstLevelModel fit function of their respective model.
    """
    # check arguments
    img_filters = img_filters if img_filters else []
    if not isinstance(dataset_path, str):
        raise TypeError('dataset_path must be a string, instead %s was given' %
                        type(task_label))
    if not os.path.exists(dataset_path):
        raise ValueError('given path do not exist: %s' % dataset_path)
    if not isinstance(task_label, str):
        raise TypeError('task_label must be a string, instead %s was given' %
                        type(task_label))
    if space_label is not None and not isinstance(space_label, str):
        raise TypeError('space_label must be a string, instead %s was given' %
                        type(space_label))
    if not isinstance(img_filters, list):
        raise TypeError('img_filters must be a list, instead %s was given' %
                        type(img_filters))
    for img_filter in img_filters:
        if (not isinstance(img_filter[0], str) or
                not isinstance(img_filter[1], str)):
            raise TypeError('filters in img filters must be (str, str), '
                            'instead %s was given' % type(img_filter))
        if img_filter[0] not in ['acq', 'ce', 'dir', 'rec', 'run', 'echo', 'desc', 'res', 'den']:
            raise ValueError("field %s is not a possible filter. Only "
                             "'acq', 'ce', 'dir', 'rec', 'run', 'echo', 'desc', 'res', 'den' "
                             "are allowed." % img_filter[0])

    # check derivatives folder is present
    derivatives_path = os.path.join(dataset_path, derivatives_folder)
    if not os.path.exists(derivatives_path):
        raise ValueError('derivatives folder does not exist in given dataset')

    # Get acq specs for models. RepetitionTime and SliceTimingReference.
    # Throw warning if no bold.json is found
    if t_r is not None:
        warn('RepetitionTime given in model_init as %d' % t_r)
        warn('slice_time_ref is %d percent of the repetition '
             'time' % slice_time_ref)
    else:
        filters = [('task', task_label)]
        for img_filter in img_filters:
            if img_filter[0] in ['acq', 'rec', 'run']:
                filters.append(img_filter)

        img_specs = get_bids_files(derivatives_path, modality_folder='func',
                                   file_tag='bold', file_type='json',
                                   filters=filters)
        # If we dont find the parameter information in the derivatives folder
        # we try to search in the raw data folder
        if not img_specs:
            img_specs = get_bids_files(dataset_path, modality_folder='func',
                                       file_tag='bold', file_type='json',
                                       filters=filters)
        if not img_specs:
            warn('No bold.json found in derivatives folder or'
                 ' in dataset folder. t_r can not be inferred and will need to'
                 ' be set manually in the list of models, otherwise their fit '
                 'will throw an exception')
        else:
            specs = json.load(open(img_specs[0], 'r'))
            if 'RepetitionTime' in specs:
                t_r = float(specs['RepetitionTime'])
            else:
                warn('RepetitionTime not found in file %s. t_r can not be '
                     'inferred and will need to be set manually in the '
                     'list of models. Otherwise their fit will throw an '
                     ' exception' % img_specs[0])
            if 'SliceTimingRef' in specs:
                slice_time_ref = float(specs['SliceTimingRef'])
            else:
                warn('SliceTimingRef not found in file %s. It will be assumed'
                     ' that the slice timing reference is 0.0 percent of the '
                     'repetition time. If it is not the case it will need to '
                     'be set manually in the generated list of models' %
                     img_specs[0])

    # Infer subjects in dataset
    sub_folders = glob.glob(os.path.join(derivatives_path, 'sub-*/'))
    sub_labels = [os.path.basename(s[:-1]).split('-')[1] for s in sub_folders]
    sub_labels = sorted(list(set(sub_labels)))

    # Build fit_kwargs dictionaries to pass to their respective models fit
    # Events and confounds files must match number of imgs (runs)
    models = []
    models_run_imgs = []
    models_events = []
    models_confounds = []
    for sub_label in sub_labels:
        # Create model
        model = FirstLevelModel(
            t_r=t_r, slice_time_ref=slice_time_ref, hrf_model=hrf_model,
            drift_model=drift_model, high_pass=high_pass,
            drift_order=drift_order, fir_delays=fir_delays,
            min_onset=min_onset, mask_img=mask_img, target_affine=target_affine,
            target_shape=target_shape, smoothing_fwhm=smoothing_fwhm,
            memory=memory, memory_level=memory_level, standardize=standardize,
            signal_scaling=signal_scaling, noise_model=noise_model,
            verbose=verbose, n_jobs=n_jobs, minimize_memory=minimize_memory,
            subject_label=sub_label)
        models.append(model)

        # Get preprocessed imgs
        if space_label is None:
            filters = [('task', task_label)] + img_filters
        else:
            filters = [('task', task_label), ('space', space_label)] + img_filters
        imgs = get_bids_files(derivatives_path, modality_folder='func',
                              file_tag='bold', file_type='nii*',
                              sub_label=sub_label, filters=filters)
        # If there is more than one file for the same (ses, run), likely we
        # have an issue of underspecification of filters.
        run_check_list = []
        # If more than one run is present the run field is mandatory in BIDS
        # as well as the ses field if more than one session is present.
        if len(imgs) > 1:
            for img in imgs:
                img_dict = parse_bids_filename(img)
                if ('_ses-' in img_dict['file_basename'] and
                        '_run-' in img_dict['file_basename']):
                    if (img_dict['ses'], img_dict['run']) in run_check_list:
                        raise ValueError(
                            'More than one nifti image found for the same run '
                            '%s and session %s. Please verify that the '
                            'desc_label and space_label labels '
                            'corresponding to the BIDS spec '
                            'were correctly specified.' %
                            (img_dict['run'], img_dict['ses']))
                    else:
                        run_check_list.append((img_dict['ses'],
                                              img_dict['run']))

                elif '_ses-' in img_dict['file_basename']:
                    if img_dict['ses'] in run_check_list:
                        raise ValueError(
                            'More than one nifti image found for the same ses '
                            '%s, while no additional run specification present'
                            '. Please verify that the desc_label and '
                            'space_label labels '
                            'corresponding to the BIDS spec '
                            'were correctly specified.' %
                            img_dict['ses'])
                    else:
                        run_check_list.append(img_dict['ses'])

                elif '_run-' in img_dict['file_basename']:
                    if img_dict['run'] in run_check_list:
                        raise ValueError(
                            'More than one nifti image found for the same run '
                            '%s. Please verify that the desc_label and '
                            'space_label labels '
                            'corresponding to the BIDS spec '
                            'were correctly specified.' %
                            img_dict['run'])
                    else:
                        run_check_list.append(img_dict['run'])
        models_run_imgs.append(imgs)

        # Get events and extra confounds
        filters = [('task', task_label)]
        for img_filter in img_filters:
            if img_filter[0] in ['acq', 'rec', 'run']:
                filters.append(img_filter)

        # Get events files
        events = get_bids_files(dataset_path, modality_folder='func',
                                file_tag='events', file_type='tsv',
                                sub_label=sub_label, filters=filters)
        if events:
            if len(events) != len(imgs):
                raise ValueError('%d events.tsv files found for %d bold '
                                 'files. Same number of event files as '
                                 'the number of runs is expected' %
                                 (len(events), len(imgs)))
            events = [pd.read_csv(event, sep='\t', index_col=None)
                      for event in events]
            models_events.append(events)
        else:
            raise ValueError('No events.tsv files found')

        # Get confounds. If not found it will be assumed there are none.
        # If there are confounds, they are assumed to be present for all runs.
        confounds = get_bids_files(derivatives_path, modality_folder='func',
                                   file_tag='desc-confounds_regressors', file_type='tsv',
                                   sub_label=sub_label, filters=filters)

        if confounds:
            if len(confounds) != len(imgs):
                raise ValueError('%d confounds.tsv files found for %d bold '
                                 'files. Same number of confound files as '
                                 'the number of runs is expected' %
                                 (len(events), len(imgs)))
            confounds = [pd.read_csv(c, sep='\t', index_col=None)
                         for c in confounds]
            models_confounds.append(confounds)

    return models, models_run_imgs, models_events, models_confounds
