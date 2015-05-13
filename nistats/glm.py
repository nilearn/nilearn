# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module presents an interface to use the glm implemented in
nipy.algorithms.statistics.models.regression.

It contains the GLM and contrast classes that are meant to be the main objects
of fMRI data analyses.

It is important to note that the GLM is meant as a one-session General Linear
Model. But inference can be performed on multiple sessions by computing fixed
effects on contrasts

Examples
--------

>>> import numpy as np
>>> from nipy.modalities.fmri.glm import GeneralLinearModel
>>> n, p, q = 100, 80, 10
>>> X, Y = np.random.randn(p, q), np.random.randn(p, n)
>>> cval = np.hstack((1, np.zeros(9)))
>>> model = GeneralLinearModel(X)
>>> model.fit(Y)
>>> z_vals = model.contrast(cval).z_score() # z-transformed statistics

Example of fixed effects statistics across two contrasts

>>> cval_ = cval.copy()
>>> np.random.shuffle(cval_)
>>> z_ffx = (model.contrast(cval) + model.contrast(cval_)).z_score()
"""

import numpy as np

from warnings import warn

import scipy.stats as sps

from nibabel import load, Nifti1Image

from nipy.labs.mask import compute_mask_sessions
from nipy.algorithms.statistics.models.regression import OLSModel, ARModel
from nipy.algorithms.statistics.utils import multiple_mahalanobis, z_score
from nipy.core.api import is_image

from nipy.testing.decorators import skip_doctest_if
from nipy.utils import HAVE_EXAMPLE_DATA

DEF_TINY = 1e-50
DEF_DOFMAX = 1e10


def data_scaling(Y):
    """Scaling of the data to have pourcent of baseline change columnwise

    Parameters
    ----------
    Y: array of shape(n_time_points, n_voxels)
       the input data

    Returns
    -------
    Y: array of shape (n_time_points, n_voxels),
       the data after mean-scaling, de-meaning and multiplication by 100
    mean : array of shape (n_voxels,)
        the data mean
    """
    mean = Y.mean(0)
    Y = 100 * (Y / mean - 1)
    return Y, mean


class GeneralLinearModel(object):
    """ This class handles the so-called on General Linear Model

    Most of what it does in the fit() and contrast() methods
    fit() performs the standard two-step ('ols' then 'ar1') GLM fitting
    contrast() returns a contrast instance, yileding statistics and p-values.
    The link between fit() and constrast is done vis the two class members:

    glm_results : dictionary of nipy.algorithms.statistics.models.
                 regression.RegressionResults instances,
                 describing results of a GLM fit

    labels : array of shape(n_voxels),
            labels that associate each voxel with a results key
    """

    def __init__(self, X):
        """
        Parameters
        ----------
        X : array of shape (n_time_points, n_regressors)
           the design matrix
        """
        self.X = X
        self.labels_ = None
        self.results_ = None

    def fit(self, Y, model='ar1', steps=100):
        """GLM fitting of a dataset using 'ols' regression or the two-pass

        Parameters
        ----------
        Y : array of shape(n_time_points, n_samples)
            the fMRI data
        model : {'ar1', 'ols'}, optional
            the temporal variance model. Defaults to 'ar1'
        steps : int, optional
            Maximum number of discrete steps for the AR(1) coef histogram
        """
        if model not in ['ar1', 'ols']:
            raise ValueError('Unknown model')

        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        if Y.shape[0] != self.X.shape[0]:
            raise ValueError('Response and predictors are inconsistent')

        # fit the OLS model
        ols_result = OLSModel(self.X).fit(Y)

        # compute and discretize the AR1 coefs
        ar1 = ((ols_result.resid[1:] * ols_result.resid[:-1]).sum(0) /
               (ols_result.resid ** 2).sum(0))
        ar1 = (ar1 * steps).astype(np.int) * 1. / steps

        # Fit the AR model acccording to current AR(1) estimates
        if model == 'ar1':
            self.results_ = {}
            self.labels_ = ar1
            # fit the model
            for val in np.unique(self.labels_):
                m = ARModel(self.X, val)
                self.results_[val] = m.fit(Y[:, self.labels_ == val])
        else:
            self.labels_ = np.zeros(Y.shape[1])
            self.results_ = {0.0: ols_result}

    def get_beta(self, column_index=None):
        """Acessor for the best linear unbiased estimated of model parameters

        Parameters
        ----------
        column_index: int or array-like of int or None, optional
            The indexed of the columns to be returned.  if None (default
            behaviour), the whole vector is returned

        Returns
        -------
        beta: array of shape (n_voxels, n_columns)
            the beta
        """
        # make colum_index a list if it an int
        if column_index == None:
            column_index = np.arange(self.X.shape[1])
        if not hasattr(column_index, '__iter__'):
            column_index = [int(column_index)]
        n_beta = len(column_index)

        # build the beta array
        beta = np.zeros((n_beta, self.labels_.size), dtype=np.float)
        for l in self.results_.keys():
            beta[:, self.labels_ == l] = self.results_[l].theta[column_index]
        return beta

    def get_mse(self):
        """Acessor for the mean squared error of the model

        Returns
        -------
        mse: array of shape (n_voxels)
            the sum of square error per voxel
        """
        # build the beta array
        mse = np.zeros(self.labels_.size, dtype=np.float)
        for l in self.results_.keys():
            mse[self.labels_ == l] = self.results_[l].MSE
        return mse

    def get_logL(self):
        """Acessor for the log-likelihood of the model

        Returns
        -------
        logL: array of shape (n_voxels,)
            the sum of square error per voxel
        """
        # build the beta array
        logL = np.zeros(self.labels_.size, dtype=np.float)
        for l in self.results_.keys():
            logL[self.labels_ == l] = self.results_[l].logL
        return logL

    def contrast(self, con_val, contrast_type=None):
        """ Specify and estimate a linear contrast

        Parameters
        ----------
        con_val : numpy.ndarray of shape (p) or (q, p)
            where q = number of contrast vectors and p = number of regressors
        contrast_type : {None, 't', 'F' or 'tmin-conjunction'}, optional
            type of the contrast.  If None, then defaults to 't' for 1D
            `con_val` and 'F' for 2D `con_val`

        Returns
        -------
        con: Contrast instance
        """
        if self.labels_ == None or self.results_ == None:
            raise ValueError('The model has not been estimated yet')
        con_val = np.asarray(con_val)
        if con_val.ndim == 1:
            dim = 1
        else:
            dim = con_val.shape[0]
        if contrast_type is None:
            if dim == 1:
                contrast_type = 't'
            else:
                contrast_type = 'F'
        if contrast_type not in ['t', 'F', 'tmin-conjunction']:
            raise ValueError('Unknown contrast type: %s' % contrast_type)

        effect_ = np.zeros((dim, self.labels_.size), dtype=np.float)
        var_ = np.zeros((dim, dim, self.labels_.size), dtype=np.float)
        if contrast_type == 't':
            for l in self.results_.keys():
                resl = self.results_[l].Tcontrast(con_val)
                effect_[:, self.labels_ == l] = resl.effect.T
                var_[:, :, self.labels_ == l] = (resl.sd ** 2).T
        else:
            for l in self.results_.keys():
                resl = self.results_[l].Fcontrast(con_val)
                effect_[:, self.labels_ == l] = resl.effect
                var_[:, :, self.labels_ == l] = resl.covariance

        dof_ = self.results_[l].df_resid
        return Contrast(effect=effect_, variance=var_, dof=dof_,
                        contrast_type=contrast_type)


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
        ==========
        effect: array of shape (contrast_dim, n_voxels)
                the effects related to the contrast
        variance: array of shape (contrast_dim, contrast_dim, n_voxels)
                  the associated variance estimate
        dof: scalar, the degrees of freedom
        contrast_type: string to be chosen among 't' and 'F'
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
            print 'Automatically converted multi-dimensional t to F contrast'
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
        ==========
        baseline: float, optional,
                  Baseline value for the test statistic
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
            stat = (multiple_mahalanobis(self.effect - baseline, 
                                          self.variance) / self.dim)
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
        ==========
        baseline: float, optional,
        Baseline value for the test statistic
        """
        if self.stat_ == None or not self.baseline == baseline:
            self.stat_ = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.contrast_type in ['t', 'tmin-conjunction']:
            p = sps.t.sf(self.stat_, np.minimum(self.dof, self.dofmax))
        elif self.contrast_type == 'F':
            p = sps.f.sf(self.stat_, self.dim, np.minimum(
                    self.dof, self.dofmax))
        else:
            raise ValueError('Unknown statistic type')
        self.p_value_ = p
        return p

    def z_score(self, baseline=0.0):
        """Return a parametric estimation of the z-score associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ==========
        baseline: float, optional,
                  Baseline value for the test statistic
        """
        if self.p_value_ == None or not self.baseline == baseline:
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


class FMRILinearModel(object):
    """ This class is meant to handle GLMs from a higher-level perspective
    i.e. by taking images as input and output
    """

    @skip_doctest_if(not HAVE_EXAMPLE_DATA)
    def __init__(self, fmri_data, design_matrices, mask='compute',
                 m=0.2, M=0.9, threshold=.5):
        """Load the data

        Parameters
        ----------
        fmri_data : Image or str or sequence of Images / str
            fmri images / paths of the (4D) fmri images
        design_matrices : arrays or str or sequence of arrays / str
            design matrix arrays / paths of .npz files
        mask : str or Image or None, optional
            string can be 'compute' or a path to an image
            image is an input (assumed binary) mask image(s),
            if 'compute', the mask is computed
            if None, no masking will be applied
        m, M, threshold: float, optional
            parameters of the masking procedure.  Should be within [0, 1]

        Notes
        -----
        The only computation done here is mask computation (if required)

        Examples
        --------
        We need the example data package for this example

        >>> from nipy.utils import example_data
        >>> from nipy.modalities.fmri.glm import FMRILinearModel
        >>> fmri_files = [example_data.get_filename('fiac', 'fiac0', run)
        ...     for run in ['run1.nii.gz', 'run2.nii.gz']]
        >>> design_files = [example_data.get_filename('fiac', 'fiac0', run)
        ...     for run in ['run1_design.npz', 'run2_design.npz']]
        >>> mask = example_data.get_filename('fiac', 'fiac0', 'mask.nii.gz')
        >>> multi_session_model = FMRILinearModel(fmri_files, design_files, mask)
        >>> multi_session_model.fit()
        >>> z_image, = multi_session_model.contrast([np.eye(13)[1]] * 2)

        The number of voxels with p < 0.001

        >>> np.sum(z_image.get_data() > 3.09)
        671
        """
        # manipulate the arguments
        if isinstance(fmri_data, basestring) or hasattr(fmri_data, 'get_data'):
            fmri_data = [fmri_data]
        if isinstance(design_matrices, (basestring, np.ndarray)):
            design_matrices = [design_matrices]
        if len(fmri_data) != len(design_matrices):
            raise ValueError('Incompatible number of fmri runs and '
                             'design matrices were provided')
        self.fmri_data, self.design_matrices = [], []
        self.glms, self.means = [], []

        # load the fmri data
        for fmri_run in fmri_data:
            if isinstance(fmri_run, basestring):
                self.fmri_data.append(load(fmri_run))
            else:
                self.fmri_data.append(fmri_run)
        # set self.affine as the affine of the first image
        self.affine = self.fmri_data[0].get_affine()

        # load the designs
        for design_matrix in design_matrices:
            if isinstance(design_matrix, basestring):
                loaded = np.load(design_matrix)
                self.design_matrices.append(loaded[loaded.files[0]])
            else:
                self.design_matrices.append(design_matrix)

        # load the mask
        if mask == 'compute':
            mask = compute_mask_sessions(
                fmri_data, m=m, M=M, cc=1, threshold=threshold, opening=0)
            self.mask = Nifti1Image(mask.astype(np.int8), self.affine)
        elif mask == None:
            mask = np.ones(self.fmri_data[0].shape[:3]).astype(np.int8)
            self.mask = Nifti1Image(mask, self.affine)
        else:
            if isinstance(mask, basestring):
                self.mask = load(mask)
            else:
                self.mask = mask

    def fit(self, do_scaling=True, model='ar1', steps=100):
        """ Load the data, mask the data, scale the data, fit the GLM

        Parameters
        ----------
        do_scaling : bool, optional
            if True, the data should be scaled as pourcent of voxel mean
        model : string, optional,
            the kind of glm ('ols' or 'ar1') you want to fit to the data
        steps : int, optional
            in case of an ar1, discretization of the ar1 parameter
        """
        from nibabel import Nifti1Image
        # get the mask as an array
        mask = self.mask.get_data().astype(np.bool)

        self.glms, self.means = [], []
        for fmri, design_matrix in zip(self.fmri_data, self.design_matrices):
            if do_scaling:
                # scale the data
                data, mean = data_scaling(fmri.get_data()[mask].T)
            else:
                data, mean = (fmri.get_data()[mask].T,
                              fmri.get_data()[mask].T.mean(0))
            mean_data = mask.astype(np.int16)
            mean_data[mask] = mean
            self.means.append(Nifti1Image(mean_data, self.affine))
            # fit the GLM
            glm = GeneralLinearModel(design_matrix)
            glm.fit(data, model, steps)
            self.glms.append(glm)

    def contrast(self, contrasts, con_id='', contrast_type=None, output_z=True,
                 output_stat=False, output_effects=False,
                 output_variance=False):
        """ Estimation of a contrast as fixed effects on all sessions

        Parameters
        ----------
        contrasts : array or list of arrays of shape (n_col) or (n_dim, n_col)
            where ``n_col`` is the number of columns of the design matrix,
            numerical definition of the contrast (one array per run)
        con_id : str, optional
            name of the contrast
        contrast_type : {'t', 'F', 'tmin-conjunction'}, optional
            type of the contrast
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
        output_images : list of nibabel images
            The desired output images
        """
        if self.glms == []:
            raise ValueError('first run fit() to estimate the model')
        if isinstance(contrasts, np.ndarray):
            contrasts = [contrasts]
        if len(contrasts) != len(self.glms):
            raise ValueError(
                'contrasts must be a sequence of %d session contrasts' %
                len(self.glms))

        contrast_ = None
        for i, (glm, con) in enumerate(zip(self.glms, contrasts)):
            if np.all(con == 0):
                warn('Contrast for session %d is null' % i)
            elif contrast_ is None:
                contrast_ = glm.contrast(con, contrast_type)
            else:
                contrast_ = contrast_ + glm.contrast(con, contrast_type)
        if output_z or output_stat:
            # compute the contrast and stat
            contrast_.z_score()

        # Prepare the returned images
        mask = self.mask.get_data().astype(np.bool)
        do_outputs = [output_z, output_stat, output_effects, output_variance]
        estimates = ['z_score_', 'stat_', 'effect', 'variance']
        descrips = ['z statistic', 'Statistical value', 'Estimated effect',
                    'Estimated variance']
        dims = [1, 1, contrast_.dim, contrast_.dim ** 2]
        n_vox = contrast_.z_score_.size
        output_images = []
        for (do_output, estimate, descrip, dim) in zip(
            do_outputs, estimates, descrips, dims):
            if do_output:
                if dim > 1:
                    result_map = np.tile(
                        mask.astype(np.float)[:, :, :, np.newaxis], dim)
                    result_map[mask] = np.reshape(
                        getattr(contrast_, estimate).T, (n_vox, dim))
                else:
                    result_map = mask.astype(np.float)
                    result_map[mask] = np.squeeze(
                        getattr(contrast_, estimate))
                output = Nifti1Image(result_map, self.affine)
                output.get_header()['descrip'] = (
                    '%s associated with contrast %s' % (descrip, con_id))
                output_images.append(output)
        return output_images
