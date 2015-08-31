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

import numpy as np
import scipy.stats as sps

from nibabel import load, Nifti1Image

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.externals.joblib import Memory
from nilearn._utils import CacheMixin
from nilearn._utils.class_inspect import get_params
from nilearn.input_data import NiftiMasker


from nilearn.masking import compute_multi_epi_mask as compute_mask_sessions
from .regression import OLSModel, ARModel
from .utils import multiple_mahalanobis, z_score, _basestring

DEF_TINY = 1e-50
DEF_DOFMAX = 1e10


def data_scaling(Y):
    """Scaling of the data to have percent of baseline change columnwise

    Parameters
    ----------
    Y : array of shape(n_time_points, n_voxels)
       the input data

    Returns
    -------
    Y : array of shape (n_time_points, n_voxels),
       the data after mean-scaling, de-meaning and multiplication by 100

    mean : array of shape (n_voxels,)
        the data mean
    """
    mean = Y.mean(0)
    if (mean == 0).any():
        warn('Mean values of 0 observed.'
             'The data have probably been centered.'
             'Scaling might not work as expected')
    mean = np.maximum(mean, 1)
    Y = 100 * (Y / mean - 1)
    return Y, mean


def session_glm(Y, X, noise_model='ar1', bins=100):
    """ GLM fit for an fMRI data matrix

    Parameters
    ----------
    Y : array of shape(n_time_points, n_voxels)
        the fMRI data

    X : array of shape (n_time_points, n_regressors)
           the design matrix

    noise_model : {'ar1', 'ols'}, optional
        the temporal variance model. Defaults to 'ar1'

    bins : int, optional
        Maximum number of discrete bins for the AR(1) coef histogram

    Returns
    -------
    labels : array of shape (n_voxels),
        a map of values on voxels used to identify the correseponding the model

    results : dictionary,
        keys correspond to the different labels values
        values are RegressionResults instances corresponding to the voxels
    """
    if noise_model not in ['ar1', 'ols']:
        raise ValueError(' %s Unknown noise model' % noise_model)

    if Y.shape[0] != X.shape[0]:
        raise ValueError('Response and predictors are inconsistent')

    # fit the OLS model
    ols_result = OLSModel(X).fit(Y)

    # compute and discretize the AR1 coefs
    ar1 = ((ols_result.resid[1:] * ols_result.resid[:-1]).sum(0) /
           (ols_result.resid ** 2).sum(0))
    ar1 = (ar1 * bins).astype(np.int) * 1. / bins

    # Fit the AR model acccording to current AR(1) estimates
    if noise_model == 'ar1':
        results = {}
        labels = ar1
        # fit the model
        for val in np.unique(labels):
            model = ARModel(X, val)
            results[val] = model.fit(Y[:, labels == val])
    else:
        labels = np.zeros(Y.shape[1])
        results = {0.0: ols_result}
    return labels, results


def compute_contrast(labels, regression_result, con_val, contrast_type=None):
    """ compute the specified contrast given an estimated glm

    Parameters
    ----------
    labels : array of shape (n_voxels),
        a map of values on voxels used to identify the correseponding the model

    results : dictionary,
        keys correspond to the differnt labels
        values are RegressionResults instances corresponding to the voxels

    con_val : numpy.ndarray of shape (p) or (q, p)
        where q = number of contrast vectors and p = number of regressors

    contrast_type : {None, 't', 'F' or 'tmin-conjunction'}, optional
        type of the contrast.  If None, then defaults to 't' for 1D
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
        if dim == 1:
            contrast_type = 't'
        else:
            contrast_type = 'F'

    if contrast_type not in ['t', 'F']:
        raise ValueError('Unknown contrast type: %s' % contrast_type)

    effect_ = np.zeros((dim, labels.size), dtype=np.float)
    var_ = np.zeros((dim, dim, labels.size), dtype=np.float)
    if contrast_type == 't':
        for label_ in regression_result.keys():
            resl = regression_result[label_].Tcontrast(con_val)
            effect_[:, labels == label_] = resl.effect.T
            var_[:, :, labels == label_] = (resl.sd ** 2).T
    else:
        for label_ in regression_result.keys():
            resl = regression_result[label_].Fcontrast(con_val)
            effect_[:, labels == label_] = resl.effect
            var_[:, :, labels == label_] = resl.covariance

    dof_ = regression_result[label_].df_resid
    return Contrast(effect=effect_, variance=var_, dof=dof_,
                    contrast_type=contrast_type)


class FirstLevelGLM(BaseEstimator, TransformerMixin, CacheMixin):
    """ Implementation of the General Linear Model for Single-session fMRI data

    mask: Niimg-like object, instance of NiftiMasker or MultiNiftiMasker,
     optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

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

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    noise_model : {'ar1', 'ols'}, optional
        the temporal variance model. Defaults to 'ar1'

    Attributes
    ----------
    labels : array of shape (n_voxels),
        a map of values on voxels used to identify the correseponding the model

    results : dictionary,
        keys correspond to the different labels values
        values are RegressionResults instances corresponding to the voxels

    
    """

    def __init__(self, mask=None, target_affine=None, target_shape=None,
             low_pass=None, high_pass=None, t_r=None, smoothing_fwhm=None,
             memory=Memory(None), memory_level=1, standardize=False,
             verbose=1, n_jobs=1, noise_model='ar1'):
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.standardize = standardize
        self.n_jobs = n_jobs
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.smoothing_fwhm = smoothing_fwhm
        self.noise_model = noise_model

    def fit(self, imgs, design_matrices):
        """ Note: design_matrices is the design matrix !
        1. does a masker job: fMRI_data -> Y
        2. fit an ols regression to (Y, X)
        3. fit an AR(1) regression of require
        This results in an internal (labels_, regression_results_) parameters

        Parameters
        ----------
        imgs: Niimg-like object or list of Niimg-like objects,
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the GLM will be fitted. If this is a list,
            the affine is considered the same for all.
            
        design_matrices: array or list of arrays
            Models of the temporal events considered in the analysis.
            the arrays have shape (n_time_points, n_regressors)

        """
        # First, learn the mask
        if not isinstance(self.mask, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask, smoothing_fwhm=self.smoothing_fwhm,
                target_affine=self.target_affine,
                standardize=self.standardize, low_pass=self.low_pass,
                high_pass=self.high_pass, mask_strategy='epi',
                mask_args={'opening': 0}, t_r=self.t_r, memory=self.memory,
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

        # make design_matrices a list of arrays
        design_matrices_ = [X for X in design_matrices]
        if isinstance(design_matrices, (_basestring, np.ndarray)):
            design_matrices_ = [design_matrices]

        design_matrices = []
        for design_matrix in design_matrices_:
            if isinstance(design_matrix, _basestring):
                loaded = np.load(design_matrix)
                design_matrices.append(loaded[loaded.files[0]])
            else:
                design_matrices.append(design_matrix)

        # make imgs a list of Nifti1Images
        if isinstance(imgs, (Nifti1Image, _basestring)):
            imgs = [imgs]

        if len(imgs) != len(design_matrices):
            raise ValueError(
                'len(imgs) %d does not match len(design_matrices) %d'
                % (len(imgs), len(design_matrices)))

        # Loop on imgs and design matrices
        self.labels_, self.results_ = [], []
        self.masker_.fit(imgs)
        for X, img in zip(design_matrices, imgs):
            Y = self.masker_.transform(img)
            if self.standardize is False:
                Y, _ = data_scaling(Y)
            labels_, results_ = session_glm(
                Y, X, noise_model=self.noise_model, bins=100)
            self.labels_.append(labels_)
            self.results_.append(results_)
        return self

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

        contrast_type : {'t', 'F', 'tmin-conjunction'}, optional
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
        """ Fit then transform"""
        return self.fit(design_matrices, fmri_images).transform(
            con_vals, contrast_type, contrast_name, output_z=True,
            output_stat=False, output_effects=False, output_variance=False)


class Contrast(object):
    """ The contrast class handles the estimation of statistical contrasts
    on a given model: student (t), Fisher (F), conjunction (tmin-conjunction).
    The important feature is that it supports addition,
    thus opening the possibility of fixed-effects models.

    The current implementation is meant to be simple,
    and could be enhanced in the future on the computational side
    (high-dimensional F constrasts may lead to memory breakage).

    Notes
    -----
    The 'tmin-conjunction' test is the valid conjunction test discussed in:
    Nichols T, Brett M, Andersson J, Wager T, Poline JB. Valid conjunction
    inference with the minimum statistic. Neuroimage. 2005 Apr 15;25(3):653-60.
    This test gives the p-value of the z-values under the conjunction null,
    i.e. the union of the null hypotheses for all terms.
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
        # Case: tmin (conjunctions)
        elif self.contrast_type == 'tmin-conjunction':
            vdiag = self.variance.reshape([self.dim ** 2] + list(
                    self.variance.shape[2:]))[:: self.dim + 1]
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(vdiag, self.tiny))
            stat = stat.min(0)

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
        if self.contrast_type in ['t', 'tmin-conjunction']:
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
