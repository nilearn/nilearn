# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module presents an interface to use the glm implemented in
nistats.regression.

It contains the GLM and contrast classes that are meant to be the main objects
of fMRI data analyses.

It is important to note that the GLM is meant as a one-session General Linear
Model. But inference can be performed on multiple sessions by computing fixed
effects on contrasts

"""

from warnings import warn
import time
import sys

import numpy as np
import pandas as pd
from nibabel import Nifti1Image

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.externals.joblib import Memory
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils import CacheMixin
from nilearn._utils.class_inspect import get_params
from nilearn.input_data import NiftiMasker
from sklearn.externals.joblib import Parallel, delayed, cpu_count

from .regression import OLSModel, ARModel
from .design_matrix import make_design_matrix
from .contrasts import compute_contrast
from .utils import _basestring


def percent_mean_scaling(Y, axis=0):
    """Scaling of the data to have percent of baseline change columnwise

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


class GroupIterator(object):
    """Group iterator

    Provides group of features for search_light loop
    that may be used with Parallel.

    Parameters
    ----------
    n_features : int
        Total number of features

    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'. Defaut is 1
    """
    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        for list_i in split:
            yield list_i


def _minimize_memory_regression_results(results):
    del results.Y
    del results.model
    del results.wY
    del results.wresid
    return results


@profile
def _group_iter_run_glm(Y, X, noise_model, bins, minimize_memory, thread_id):
    """Function for grouped iterations of given function

    Parameters
    -----------
    Y : array of shape (n_time_points, n_voxels)
        The fMRI data.

    X : array of shape (n_time_points, n_regressors)
        The design matrix.

    noise_model : {'ar1', 'ols'}, optional
        The temporal variance model. Defaults to 'ar1'.

    bins : int, optional
        Maximum number of discrete bins for the AR(1) coef histogram.

    Returns
    -------
    labels : array of shape (n_voxels,),
        A map of values on voxels used to identify the corresponding model.

    results : dict,
        Keys correspond to the different labels values
        values are RegressionResults instances corresponding to the voxels.
    """
    # fit the OLS model
    ols_result = OLSModel(X).fit(Y)

    if noise_model == 'ar1':
        # compute and discretize the AR1 coefs
        ar1 = ((ols_result.resid[1:] * ols_result.resid[:-1]).sum(axis=0) /
               (ols_result.resid ** 2).sum(axis=0))
        del ols_result
        ar1 = (ar1 * bins).astype(np.int) * 1. / bins
        # Fit the AR model acccording to current AR(1) estimates
        results = {}
        labels = ar1 + (thread_id * 1000)
        # fit the model
        for val in np.unique(ar1):
            model = ARModel(X, val)
            key = val + (thread_id * 1000)
            results[key] = model.fit(Y[:, labels == key])
            if minimize_memory:
                results[key] = _minimize_memory_regression_results(results[key])
        del ar1
    else:
        labels = np.zeros(Y.shape[1]) + (thread_id * 1000)
        results = {0.0 + (thread_id * 1000): ols_result}
    return labels, results


@profile
def run_glm(Y, X, noise_model='ar1', bins=100, n_jobs=1, verbose=0,
            minimize_memory=False):
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

    n_voxels = Y.shape[1]
    group_iter = GroupIterator(n_voxels, n_jobs)

    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_group_iter_run_glm)(Y[:, voxels], X, noise_model, bins,
                                     minimize_memory, thread_id)
        for thread_id, voxels in enumerate(group_iter))

    labels, reg_res = zip(*res)
    labels = np.concatenate(labels)
    results = {}
    for reg in reg_res:
        results.update(reg)

    return labels, results


def _check_list_length_match(list_1, list_2, var_name_1, var_name_2):
    if len(list_1) != len(list_2):
        raise ValueError(
            'len(%s) %d does not match len(%s) %d'
            % (var_name_1, len(list_1), var_name_2, len(list_2)))


def _check_and_load_tables(tables_, var_name):
    tables = []
    for table_idx, table in enumerate(tables_):
        if isinstance(table, _basestring):
            loaded = pd.read_csv(table, index_col=0)
            tables.append(loaded)
        elif isinstance(table, pd.DataFrame):
            tables.append(table)
        else:
            raise TypeError('%s can only be a pandas DataFrames or a'
                            'string. A %s was provided at idx %d' %
                            (var_name, type(table), table_idx))
    return tables


def _check_run_tables(run_imgs, tables_, tables_name):
    if isinstance(tables_, (_basestring, pd.DataFrame)):
        tables_ = [tables_]
    _check_list_length_match(run_imgs, tables_, 'run_imgs', tables_name)
    tables_ = _check_and_load_tables(tables_, tables_name)
    return tables_


class FirstLevelModel(BaseEstimator, TransformerMixin, CacheMixin):
    """ Implementation of the General Linear Model for Single-session fMRI data

    Parameters
    ----------

    t_r: float
        This parameter indicates repetition times of the experimental runs.
        It is necessary to correctly consider times in the design matrix.
        This parameter is also passed to nilearn.signal.clean.
        Please see the related documentation for details.

    slice_time_ref: float, optional (default 0.)
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
        model.

    min_onset : float, optional
        This parameter specifies the minimal onset relative to the design
        (in seconds). Events that start before (slice_time_ref * t_r +
        min_onset) are not considered.

    mask: Niimg-like, NiftiMasker or MultiNiftiMasker object, optional,
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to nilearn.image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to nilearn.image.resample_img. Please see the
        related documentation for details.

    low_pass: False or float, optional
        This parameter is passed to nilearn.signal.clean.
        Please see the related documentation for details.

    high_pass: False or float, optional
        This parameter is passed to nilearn.signal.clean.
        Please see the related documentation for details.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    memory: string, optional
        Path to the directory used to cache the masking process and the glm
        fit. By default, no caching is done. Creates instance of joblib.Memory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    percent_signal_change: bool, optional,
        If True, fMRI signals are scaled to percent of the mean value
        Incompatible with standardize (standardize=False is enforced when\
        percent_signal_change is True).

    noise_model : {'ar1', 'ols'}, optional
        The temporal variance model. Defaults to 'ar1'

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    Attributes
    ----------
    labels : array of shape (n_voxels,),
        a map of values on voxels used to identify the corresponding model

    results : dict,
        with keys corresponding to the different labels values
        values are RegressionResults instances corresponding to the voxels
    """

    def __init__(self, t_r=None, slice_time_ref=None, hrf_model='canonical',
                 drift_model='cosine', period_cut=128, drift_order=1,
                 fir_delays=[0], min_onset=-24, mask=None, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None,
                 smoothing_fwhm=None, memory=None, memory_level=1,
                 standardize=False, percent_signal_change=True,
                 noise_model='ar1', verbose=1, n_jobs=1,
                 minimize_memory=False):
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
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.smoothing_fwhm = smoothing_fwhm
        self.memory = Memory(memory)
        self.memory_level = memory_level
        self.standardize = standardize
        self.percent_signal_change = percent_signal_change
        if self.percent_signal_change:
            self.standardize = False
        self.noise_model = noise_model
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.minimize_memory = minimize_memory

    def get_model_parameters(self, ignore=None):
        """Get model parameters as a dictionary.

        Parameters
        ----------
        ignore: None or list of strings, optional
            Names of the parameters that are not returned.

        Returns
        -------
        params: dict
            The dict of parameters
        """
        return get_params(FirstLevelModel, self, ignore=ignore)

    @profile
    def fit(self, run_imgs, paradigms=None, confounds=None,
            design_matrices=None):
        """ Fit the GLM

        For each run:
        1. create design matrix
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        run_imgs: Niimg-like object or list of Niimg-like objects,
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the GLM will be fitted. If this is a list,
            the affine is considered the same for all.

        paradigms: pandas Dataframe or string or list of pandas DataFrames or
                   strings,
            fMRI paradigms used to build design matrices. One paradigm expected
            per run_img. Ignored in case designs is not None.

        confounds: pandas Dataframe or string or list of pandas DataFrames or
                   strings,
            Each column in a DataFrame corresponds to a confound variable
            to be included in the regression model of the respective run_img.
            The number of rows must match the number of volumes in the
            respective run_img. Ignored in case designs is not None.

        design_matrices: pandas DataFrame or list of pandas DataFrames,
            Design matrices that will be used to fit the GLM.
        """
        # Check arguments
        if isinstance(run_imgs, (_basestring, Nifti1Image)):
            run_imgs = [run_imgs]

        if design_matrices is None:
            if paradigms is None:
                raise ValueError('paradigms or design matrices must be provided')
            if self.t_r is None:
                raise ValueError('t_r not given to FirstLevelModel object'
                                 'to compute design from paradigm')
            if self.slice_time_ref is None:
                raise ValueError('slice_time_ref not given to FirstLevelModel'
                                 'object to compute design from paradigm')
        else:
            design_matrices = _check_run_tables(run_imgs, design_matrices,
                                                'design_matrices')

        if paradigms is not None:
            paradigms = _check_run_tables(run_imgs, paradigms, 'paradigms')

        if confounds is not None:
            confounds = _check_run_tables(run_imgs, confounds, 'confounds')

        # Learn the mask
        if not isinstance(self.mask, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask, smoothing_fwhm=self.smoothing_fwhm,
                target_affine=self.target_affine,
                standardize=self.standardize, low_pass=self.low_pass,
                high_pass=self.high_pass, mask_strategy='epi',
                t_r=self.t_r, memory=self.memory,
                verbose=max(0, self.verbose - 1),
                target_shape=self.target_shape,
                memory_level=self.memory_level)
        else:
            self.masker_ = clone(self.mask)
            for param_name in ['target_affine', 'target_shape',
                               'smoothing_fwhm', 'low_pass', 'high_pass',
                               't_r', 'memory', 'memory_level']:
                our_param = getattr(self, param_name)
                if our_param is None:
                    continue
                if getattr(self.masker_, param_name) is not None:
                    warn('Parameter %s of the masker overriden' % param_name)
                setattr(self.masker_, param_name, our_param)
        self.masker_.fit(run_imgs[0])

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
                    remaining = 'estimating time remaining'
                else:
                    remaining = (100. - percent) / max(0.01, percent) * dt
                    remaining = '%i seconds remaining' % remaining
                sys.stderr.write(
                    "Computing run %d out of %d runs (%s)\r"
                    % (run_idx, n_runs, remaining))

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
                    confounds_names = confounds[run_idx].columns
                else:
                    confounds_matrix = None
                    confounds_names = None
                start_time = self.slice_time_ref * self.t_r
                end_time = (n_scans - self.slice_time_ref) * self.t_r
                frame_times = np.linspace(start_time, end_time, n_scans)
                design = make_design_matrix(frame_times, paradigms[run_idx],
                                            self.hrf_model, self.drift_model,
                                            self.period_cut, self.drift_order,
                                            self.fir_delays, confounds_matrix,
                                            confounds_names, self.min_onset)
            else:
                design = design_matrices[run_idx]
            self.design_matrices_.append(design)

            # Compute GLM
            @profile
            def masking(run_img):
                return self.masker_.transform(run_img)
            Y = masking(run_img)

            # Y = self.masker_.transform(run_img)
            if self.percent_signal_change:
                Y, _ = percent_mean_scaling(Y)
            if self.memory is not None:
                mem_glm = self.memory.cache(run_glm)
            else:
                mem_glm = run_glm
            labels, results = mem_glm(Y, design,
                                      noise_model=self.noise_model,
                                      bins=100, n_jobs=self.n_jobs,
                                      minimize_memory=self.minimize_memory)
            # if self.minimize_memory:
            #     for key in results:
            #         results[key] = _minimize_memory_regression_results(results[key])
            self.labels_.append(labels)
            self.results_.append(results)
            del Y

        # Report progress
        if self.verbose > 0:
            sys.stderr.write("\nComputation of %d runs done in %i seconds\n"
                             % (n_runs, time.time() - t0))

        return self

    def get_design_matrices(self):
        """Get design matrices computed during model fit

        Returns
        -------
        design_matrices : list of DataFrames,
            Holds the design matrices computed for the corresponding run_imgs
        """
        if self.design_matrices_ is None:
            raise ValueError('The model has not been fit yet')
        else:
            return self.design_matrices_

    def compute_contrast(self, contrast_def, contrast_name=None,
                         stat_type=None,
                         output_type='z_score'):
        """Generate different outputs corresponding to
        the contrasts provided e.g. z_map, t_map, effects and variance.
        In multi-session case, outputs the fixed effects map.

        Parameters
        ----------
        con_vals : array or list of arrays of shape (n_col) or (n_dim, n_col)
            where ``n_col`` is the number of columns of the design matrix,
            numerical definition of the contrast (one array per run)

        contrast_name : str, optional
            name of the contrast

        stat_type : {'t', 'F'}, optional
            type of the contrast

        output_type : str, optional
            Type of the output map. Can be 'z_score', 'stat', 'p_value',
            'effect' or 'variance'

        Returns
        -------
        output_image : Nifti1Image
            The desired output image

        """
        if self.labels_ is None or self.results_ is None:
            raise ValueError('The model has not been fit yet')

        if isinstance(contrast_def, (_basestring)):
            raise ValueError('Formulas not implemented yet')
        elif isinstance(contrast_def, (list, np.ndarray)):
            con_vals = contrast_def

        if isinstance(con_vals, np.ndarray):
            con_vals = [con_vals]
        if len(con_vals) != len(self.results_):
            raise ValueError(
                'contrasts must be a sequence of %d session contrasts' %
                len(self.results_))

        contrast = None
        for i, (labels_, results_, con_val) in enumerate(zip(
                self.labels_, self.results_, con_vals)):
            if np.all(con_val == 0):
                warn('Contrast for session %d is null' % i)
                continue
            contrast_ = compute_contrast(labels_, results_, con_val,
                                         stat_type)
            if contrast is None:
                contrast = contrast_
            else:
                contrast = contrast + contrast_

        estimate_ = getattr(contrast, output_type)()
        # Prepare the returned images
        output = self.masker_.inverse_transform(estimate_)
        if contrast_name is None:
            contrast_name = str(con_vals)
        output.get_header()['descrip'] = (
            '%s of contrast %s' % (output_type, contrast_name))
        return output
