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
import scipy.stats as sps
import pandas as pd
from nibabel import Nifti1Image

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.externals.joblib import Memory
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils import CacheMixin
from nilearn._utils.class_inspect import get_params
from nilearn.input_data import NiftiMasker
from sklearn.externals.joblib import Parallel, delayed, cpu_count

from nilearn.masking import compute_multi_epi_mask as compute_mask_sessions

from .regression import OLSModel, ARModel
from .design_matrix import make_design_matrix
from .utils import multiple_mahalanobis, z_score, _basestring

DEF_TINY = 1e-50
DEF_DOFMAX = 1e10


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


def _group_iter_session_glm(Y, X, noise_model, bins, thread_id):
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

    # compute and discretize the AR1 coefs
    ar1 = ((ols_result.resid[1:] * ols_result.resid[:-1]).sum(axis=0) /
           (ols_result.resid ** 2).sum(axis=0))
    ar1 = (ar1 * bins).astype(np.int) * 1. / bins

    # Fit the AR model acccording to current AR(1) estimates
    if noise_model == 'ar1':
        results = {}
        labels = ar1 + (thread_id * 1000)
        # fit the model
        for val in np.unique(ar1):
            model = ARModel(X, val)
            key = val + (thread_id * 1000)
            results[key] = model.fit(Y[:, labels == key])
    else:
        labels = np.zeros(Y.shape[1]) + (thread_id * 1000)
        results = {0.0 + (thread_id * 1000): ols_result}
    return labels, results


def session_glm(Y, X, noise_model='ar1', bins=100, n_jobs=1, verbose=0):
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
        delayed(_group_iter_session_glm)(Y[:, voxels], X, noise_model, bins,
                                         thread_id)
        for thread_id, voxels in enumerate(group_iter))

    labels, reg_res = zip(*res)
    labels = np.concatenate(labels)
    results = {}
    for reg in reg_res:
        results.update(reg)

    return labels, results


def compute_contrast(labels, regression_result, con_val, contrast_type=None):
    """ Compute the specified contrast given an estimated glm

    Parameters
    ----------
    labels : array of shape (n_voxels,),
        A map of values on voxels used to identify the corresponding model

    results : dict,
        With keys corresponding to the different labels
        values are RegressionResults instances corresponding to the voxels.

    con_val : numpy.ndarray of shape (p) or (q, p)
        Where q = number of contrast vectors and p = number of regressors.

    contrast_type : {None, 't', 'F'}, optional
        Type of the contrast.  If None, then defaults to 't' for 1D
        `con_val` and 'F' for 2D `con_val`

    Returns
    -------
    con : Contrast instance,
        Yields the statistics of the contrast (effects, variance, p-values)
    """
    con_val = np.asarray(con_val)
    dim = 1
    if con_val.ndim > 1:
        dim = con_val.shape[0]

    if contrast_type is None:
        contrast_type = 't' if dim == 1 else 'F'

    acceptable_contrast_types = ['t', 'F']
    if contrast_type not in acceptable_contrast_types:
        raise ValueError(
            '"{0}" is not a known contrast type. Allowed types are {1}'.
            format(contrast_type, acceptable_contrast_types))

    effect_ = np.zeros((dim, labels.size))
    var_ = np.zeros((dim, dim, labels.size))
    if contrast_type == 't':
        for label_ in regression_result:
            label_mask = labels == label_
            resl = regression_result[label_].Tcontrast(con_val)
            effect_[:, label_mask] = resl.effect.T
            var_[:, :, label_mask] = (resl.sd ** 2).T
    else:
        for label_ in regression_result:
            label_mask = labels == label_
            resl = regression_result[label_].Fcontrast(con_val)
            effect_[:, label_mask] = resl.effect
            var_[:, :, label_mask] = resl.covariance

    dof_ = regression_result[label_].df_resid
    return Contrast(effect=effect_, variance=var_, dof=dof_,
                    contrast_type=contrast_type)


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

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

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

    def __init__(self, t_r, slice_time_ref=0., hrf_model='canonical',
                 drift_model='cosine', period_cut=128, drift_order=1,
                 fir_delays=[0], min_onset=-24, mask=None, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None,
                 smoothing_fwhm=None, memory=Memory(None), memory_level=1,
                 standardize=False, percent_signal_change=True,
                 noise_model='ar1', verbose=1, n_jobs=1):
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
        self.memory = memory
        self.memory_level = memory_level
        self.standardize = standardize
        self.percent_signal_change = percent_signal_change
        if self.percent_signal_change:
            self.standardize = False
        self.noise_model = noise_model
        self.verbose = verbose
        self.n_jobs = n_jobs

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

    def fit(self, run_imgs, paradigms, confounds=None):
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
            per run_img.

        confounds: pandas Dataframe or string or list of pandas DataFrames or
                   strings,
            Each column in a DataFrame corresponds to a confound variable
            to be included in the regression model of the respective run_img.
            The number of rows must match the number of volumes in the
            respective run_img.
        """
        # Check arguments
        if isinstance(run_imgs, (_basestring, Nifti1Image)):
            run_imgs = [run_imgs]
        if isinstance(paradigms, (_basestring, pd.DataFrame)):
            paradigms = [paradigms]
        _check_list_length_match(run_imgs, paradigms, 'run_imgs', 'paradigms')
        paradigms = _check_and_load_tables(paradigms, 'paradigms')
        if confounds is not None:
            if isinstance(confounds, (_basestring, pd.DataFrame)):
                confounds = [confounds]
            _check_list_length_match(run_imgs, confounds, 'run_imgs',
                                     'confounds')
            confounds = _check_and_load_tables(confounds, 'confounds')

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
            n_scans = run_img.get_data().shape[3]
            if confounds is not None:
                confounds_matrix = confounds[run_idx].values
                if confounds_matrix.shape[0] != n_scans:
                    raise ValueError('Rows in confounds does not match n_scans'
                                     ' in run_img at index %d' % (run_idx,))
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
            self.design_matrices_.append(design)

            # Compute GLM
            Y = self.masker_.transform(run_img)
            if self.percent_signal_change:
                Y, _ = percent_mean_scaling(Y)
            labels_, results_ = session_glm(Y, design,
                                            noise_model=self.noise_model,
                                            bins=100, n_jobs=self.n_jobs)
            self.labels_.append(labels_)
            self.results_.append(results_)
            del run_img
            del Y

        # Report progress
        if self.verbose > 0:
            sys.stderr.write("Computation of %d runs done in %i seconds"
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

    def compute_cache_dir():
        pass

    def compute_contrast():
        pass

    def save():
        pass

    def summary():
        pass

    def transform(self, con_vals, contrast_type=None, contrast_name='',
                  output_z=True, output_stat=False, output_effects=False,
                  output_variance=False):
        """Generate different outputs corresponding to
        the contrasts provided e.g. z_map, t_map, effects and variance.
        In multi-session case, outputs the fixed effects map.

        Parameters
        ----------
        con_vals : array or list of arrays of shape (n_col) or (n_dim, n_col)
            where ``n_col`` is the number of columns of the design matrix,
            numerical definition of the contrast (one array per run)

        contrast_type : {'t', 'F'}, optional
            type of the contrast

        contrast_name : str, optional
            name of the contrast

        output_z : bool, optional
            Return or not the corresponding z-stat image

        output_stat : bool, optional
            Return or not the base (t/F) stat image

        output_effects : bool, optional
            Return or not the corresponding effect image

        output_variance : bool, optional
            Return or not the corresponding variance image

        Returns
        -------
        output_images : list of Nifti1Images
            The desired output images

        """
        if self.labels_ is None or self.results_ is None:
            raise ValueError('The model has not been fit yet')

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
            contrast_ = compute_contrast(labels_, results_, con_val,
                                         contrast_type)
            if contrast is None:
                contrast = contrast_
            else:
                contrast = contrast + contrast_

        if output_z or output_stat:
            # compute the contrast and stat
            contrast.z_score()

        # Prepare the returned images
        do_outputs = [output_z, output_stat, output_effects, output_variance]
        estimates = ['z_score_', 'stat_', 'effect', 'variance']
        descrips = ['z statistic', 'Statistical value', 'Estimated effect',
                    'Estimated variance']
        output_images = []
        for do_output, estimate, descrip in zip(
                do_outputs, estimates, descrips):
            if not do_output:
                continue
            estimate_ = getattr(contrast, estimate)
            if estimate_.ndim == 3:
                shape_ = estimate_.shape
                estimate_ = np.reshape(estimate_,
                                       (shape_[0] * shape_[1], shape_[2]))
            output = self.masker_.inverse_transform(estimate_)
            output.get_header()['descrip'] = (
                '%s of contrast %s' % (descrip, contrast_name))
            output_images.append(output)
        return output_images

    def fit_transform(
        self, design_matrices, fmri_images, con_vals, contrast_type=None,
        contrast_name='', output_z=True, output_stat=False,
        output_effects=False, output_variance=False):
        """ Fit then transform. For more details,
        see FirstLevelGLM.fit and FirstLevelGLM.transform documentation"""
        return self.fit(design_matrices, fmri_images).transform(
            con_vals, contrast_type, contrast_name, output_z=True,
            output_stat=False, output_effects=False, output_variance=False)





class Contrast(object):
    """ The contrast class handles the estimation of statistical contrasts
    on a given model: student (t) or Fisher (F).
    The important feature is that it supports addition,
    thus opening the possibility of fixed-effects models.

    The current implementation is meant to be simple,
    and could be enhanced in the future on the computational side
    (high-dimensional F constrasts may lead to memory breakage).
    """

    def __init__(self, effect, variance, dof=DEF_DOFMAX, contrast_type='t',
                 tiny=DEF_TINY, dofmax=DEF_DOFMAX):
        """
        Parameters
        ----------
        effect : array of shape (contrast_dim, n_voxels)
            the effects related to the contrast

        variance : array of shape (contrast_dim, contrast_dim, n_voxels)
            the associated variance estimate

        dof : scalar
            the degrees of freedom of the resiudals

        contrast_type: {'t', 'F'}
            specification of the contrast type
        """
        if variance.ndim != 3:
            raise ValueError('Variance array should have 3 dimensions')
        if effect.ndim != 2:
            raise ValueError('Variance array should have 2 dimensions')
        if variance.shape[0] != variance.shape[1]:
            raise ValueError('Inconsistent shape for the variance estimate')
        if ((variance.shape[1] != effect.shape[0]) or
            (variance.shape[2] != effect.shape[1])):
            raise ValueError('Effect and variance have inconsistent shape')

        self.effect = effect
        self.variance = variance
        self.dof = float(dof)
        self.dim = effect.shape[0]
        if self.dim > 1 and contrast_type is 't':
            print('Automatically converted multi-dimensional t to F contrast')
            contrast_type = 'F'
        self.contrast_type = contrast_type
        self.stat_ = None
        self.p_value_ = None
        self.baseline = 0
        self.tiny = tiny
        self.dofmax = dofmax

    def stat(self, baseline=0.0):
        """ Return the decision statistic associated with the test of the
        null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ----------
        baseline : float, optional
            Baseline value for the test statistic

        Returns
        -------
        stat: 1-d array, shape=(n_voxels,)
            statistical values, one per voxel
        """
        self.baseline = baseline

        # Case: one-dimensional contrast ==> t or t**2
        if self.dim == 1:
            # avoids division by zero
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, self.tiny))
            if self.contrast_type == 'F':
                stat = stat ** 2
        # Case: F contrast
        elif self.contrast_type == 'F':
            # F = |t|^2/q ,  |t|^2 = e^t inv(v) e
            if self.effect.ndim == 1:
                self.effect = self.effect[np.newaxis]
            if self.variance.ndim == 1:
                self.variance = self.variance[np.newaxis, np.newaxis]
            stat = (multiple_mahalanobis(
                    self.effect - baseline, self.variance) / self.dim)
        # Unknwon stat
        else:
            raise ValueError('Unknown statistic type')
        self.stat_ = stat
        return stat.ravel()

    def p_value(self, baseline=0.0):
        """Return a parametric estimate of the p-value associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ----------
        baseline : float, optional
            baseline value for the test statistic

        Returns
        -------
        p_values : 1-d array, shape=(n_voxels,)
            p-values, one per voxel
        """
        if self.stat_ is None or not self.baseline == baseline:
            self.stat_ = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.contrast_type == 't':
            p_values = sps.t.sf(self.stat_, np.minimum(self.dof, self.dofmax))
        elif self.contrast_type == 'F':
            p_values = sps.f.sf(self.stat_, self.dim, np.minimum(
                    self.dof, self.dofmax))
        else:
            raise ValueError('Unknown statistic type')
        self.p_value_ = p_values
        return p_values

    def z_score(self, baseline=0.0):
        """Return a parametric estimation of the z-score associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ----------
        baseline: float, optional,
                  Baseline value for the test statistic

        Returns
        -------
        z_score: 1-d array, shape=(n_voxels,)
            statistical values, one per voxel

        """
        if self.p_value_ is None or not self.baseline == baseline:
            self.p_value_ = self.p_value(baseline)

        # Avoid inf values kindly supplied by scipy.
        self.z_score_ = z_score(self.p_value_)
        return self.z_score_

    def __add__(self, other):
        """Addition of selfwith others, Yields an new Contrast instance
        This should be used only on indepndent contrasts"""
        if self.contrast_type != other.contrast_type:
            raise ValueError(
                'The two contrasts do not have consistant type dimensions')
        if self.dim != other.dim:
            raise ValueError(
                'The two contrasts do not have compatible dimensions')
        effect_ = self.effect + other.effect
        variance_ = self.variance + other.variance
        dof_ = self.dof + other.dof
        return Contrast(effect=effect_, variance=variance_, dof=dof_,
                        contrast_type=self.contrast_type)

    def __rmul__(self, scalar):
        """Multiplication of the contrast by a scalar"""
        scalar = float(scalar)
        effect_ = self.effect * scalar
        variance_ = self.variance * scalar ** 2
        dof_ = self.dof
        return Contrast(effect=effect_, variance=variance_, dof=dof_,
                        contrast_type=self.contrast_type)

    __mul__ = __rmul__

    def __div__(self, scalar):
        return self.__rmul__(1 / float(scalar))
