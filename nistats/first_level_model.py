# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module presents an interface to use the glm implemented in
nistats.regression.

It contains the GLM and contrast classes that are meant to be the main objects
of fMRI data analyses.

Author: Bertrand Thirion, Martin Perez-Guevara, 2016

"""

from warnings import warn
import time
import sys

import numpy as np
from nibabel import Nifti1Image

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.externals.joblib import Memory
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils import CacheMixin
from nilearn.input_data import NiftiMasker
from sklearn.externals.joblib import Parallel, delayed
from patsy import DesignInfo

from .regression import OLSModel, ARModel, SimpleRegressionResults
from .design_matrix import make_design_matrix
from .contrasts import _fixed_effect_contrast
from .utils import _basestring, _check_run_tables


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

    hrf_model : string, optional
        This parameter specifies the hemodynamic response function (HRF) for
        the design matrices. It can be 'canonical', 'canonical with derivative'
        or 'fir'.

    drift_model : string, optional
        This parameter specifies the desired drift model for the design
        matrices. It can be 'polynomial', 'cosine' or 'blank'.

    period_cut : float, optional
        This parameter specifies the cut period of the low-pass filter in
        seconds for the design matrices.

    drift_order : int, optional
        This parameter specifices the order of the drift model (in case it is
        polynomial) for the design matrices.

    fir_delays : array of shape(n_onsets) or list, optional
        In case of FIR design, yields the array of delays used in the FIR
        model, in seconds.

    min_onset : float, optional
        This parameter specifies the minimal onset relative to the design
        (in seconds). Events that start before (slice_time_ref * t_r +
        min_onset) are not considered.

    mask : Niimg-like, NiftiMasker object or False, optional
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

    subject_id : string, optional
        This id will be used to identify a `FirstLevelModel` when passed to
        a `SecondLevelModel` object.

    Attributes
    ----------
    labels : array of shape (n_voxels,),
        a map of values on voxels used to identify the corresponding model

    results : dict,
        with keys corresponding to the different labels values
        values are RegressionResults instances corresponding to the voxels

    """
    def __init__(self, t_r=None, slice_time_ref=0., hrf_model='glover',
                 drift_model='cosine', period_cut=128, drift_order=1,
                 fir_delays=[0], min_onset=-24, mask=None, target_affine=None,
                 target_shape=None, smoothing_fwhm=None, memory=Memory(None),
                 memory_level=1, standardize=False, signal_scaling=0,
                 noise_model='ar1', verbose=0, n_jobs=1,
                 minimize_memory=True, subject_id=None):
        # design matrix parameters
        self.t_r = t_r
        self.slice_time_ref = slice_time_ref
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.period_cut = period_cut
        self.drift_order = drift_order
        self.fir_delays = fir_delays
        self.min_onset = min_onset
        # glm parameters
        self.mask = mask
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.smoothing_fwhm = smoothing_fwhm
        if isinstance(memory, _basestring):
            self.memory = Memory(memory)
        else:
            self.memory = memory
        self.memory_level = memory_level
        self.standardize = standardize
        if signal_scaling in [0, 1, (0, 1)]:
            self.scaling_axis = signal_scaling
            self.signal_scaling = True
            self.standardize = False
        elif signal_scaling is False:
            self.signal_scaling = signal_scaling
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
        self.subject_id = subject_id

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
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the GLM will be fitted. If this is a list,
            the affine is considered the same for all.

        events: pandas Dataframe or string or list of pandas DataFrames or
                   strings,
            fMRI events used to build design matrices. One events object
            expected per run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        confounds: pandas Dataframe or string or list of pandas DataFrames or
                   strings,
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
        if not isinstance(run_imgs, (list, tuple)):
            run_imgs = [run_imgs]
        for rimg in run_imgs:
            if not isinstance(rimg, (_basestring, Nifti1Image)):
                raise ValueError('run_imgs must be Niimg-like object or list'
                                 ' of Niimg-like objects')
        # check all information necessary to build design matrices is available
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
        if self.mask is False:
            # We create a dummy mask to preserve functionality of api
            ref_img = check_niimg(run_imgs[0])
            self.mask = Nifti1Image(np.ones(ref_img.shape[:3]),
                                    ref_img.get_affine())
        if not isinstance(self.mask, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask, smoothing_fwhm=self.smoothing_fwhm,
                target_affine=self.target_affine,
                standardize=self.standardize, mask_strategy='epi',
                t_r=self.t_r, memory=self.memory,
                verbose=max(0, self.verbose - 2),
                target_shape=self.target_shape,
                memory_level=self.memory_level)
            self.masker_.fit(run_imgs[0])
        else:
            if self.mask.mask_img_ is None and self.masker_ is None:
                self.masker_ = clone(self.mask)
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
                self.masker_ = self.mask

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
                n_scans = run_img.get_data().shape[3]
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
                design = make_design_matrix(frame_times, events[run_idx],
                                            self.hrf_model, self.drift_model,
                                            self.period_cut, self.drift_order,
                                            self.fir_delays, confounds_matrix,
                                            confounds_names, self.min_onset)
            else:
                design = design_matrices[run_idx]
            self.design_matrices_.append(design)

            # Mask and prepare data for GLM
            if self.verbose > 1:
                t_masking = time.time()
                sys.stderr.write('Starting masker computation')
            Y = self.masker_.transform(run_img)
            if self.verbose > 1:
                t_masking = time.time() - t_masking
                sys.stderr.write('Masker took %d seconds          \n' % t_masking)

            if self.signal_scaling:
                Y, _ = mean_scaling(Y, self.scaling_axis)
            if self.memory is not None:
                mem_glm = self.memory.cache(run_glm, ignore=['n_jobs'])
            else:
                mem_glm = run_glm

            # compute GLM
            if self.verbose > 1:
                t_glm = time.time()
                sys.stderr.write('Performing GLM computation\r')
            labels, results = mem_glm(Y, design.as_matrix(),
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
            the linear constraint of the Patsy library. Basically one can use
            the name of the conditions as they appear in the design matrix of
            the fitted model combined with operators /*+- and numbers.
            Please checks the patsy documentation for formula examples:
            http://patsy.readthedocs.io/en/latest/API-reference.html#patsy.DesignInfo.linear_constraint

        stat_type : {'t', 'F'}, optional
            type of the contrast

        output_type : str, optional
            Type of the output map. Can be 'z_score', 'stat', 'p_value',
            'effect_size' or 'effect_variance'

        Returns
        -------
        output : Nifti1Image
            The desired output image

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

        # Translate formulas to vectors with patsy
        design_info = DesignInfo(self.design_matrices_[0].columns.tolist())
        for cidx, con in enumerate(con_vals):
            if not isinstance(con, np.ndarray):
                con_vals[cidx] = design_info.linear_constraint(con).coefs

        n_runs = len(self.labels_)
        if len(con_vals) != n_runs:
            warn('One contrast given, assuming it for all %d runs' % n_runs)
            con_vals = con_vals * n_runs
        if isinstance(output_type, _basestring):
            if output_type not in ['z_score', 'stat', 'p_value', 'effect_size',
                                   'effect_variance']:
                raise ValueError('output_type must be one of "z_score", "stat",'
                                 ' "p_value","effect_size" or "effect_variance"')
        else:
            raise ValueError('output_type must be one of "z_score", "stat",'
                             ' "p_value","effect_size" or "effect_variance"')

        contrast = _fixed_effect_contrast(self.labels_, self.results_,
                                          con_vals, stat_type)

        estimate_ = getattr(contrast, output_type)()
        # Prepare the returned images
        output = self.masker_.inverse_transform(estimate_)
        contrast_name = str(con_vals)
        output.get_header()['descrip'] = (
            '%s of contrast %s' % (output_type, contrast_name))

        return output
