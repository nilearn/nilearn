"""
This module implement classes to handle statistical tests on likelihood models

Author: Bertrand Thirion, 2011--2015
"""

import numpy as np

from nibabel.onetime import setattr_on_read
from scipy.linalg import inv
from scipy.stats import t as t_distribution

from .utils import positive_reciprocal

# Inverse t cumulative distribution
inv_t_cdf = t_distribution.ppf

class LikelihoodModelResults(object):
    ''' Class to contain results from likelihood models '''

    # This is the class in which things like AIC, BIC, llf can be implemented as
    # methods, not computed in, say, the fit method of OLSModel

    def __init__(self, theta, Y, model, cov=None, dispersion=1., nuisance=None,
                 rank=None):
        ''' Set up results structure

        Parameters
        ----------
        theta : ndarray
            parameter estimates from estimated model

        Y : ndarray
            data

        model : ``LikelihoodModel`` instance
            model used to generate fit

        cov : None or ndarray, optional
            covariance of thetas

        dispersion : scalar, optional
            multiplicative factor in front of `cov`

        nuisance : None of ndarray
            parameter estimates needed to compute logL

        rank : None or scalar
            rank of the model.  If rank is not None, it is used for df_model
            instead of the usual counting of parameters.

        Notes
        -----
        The covariance of thetas is given by:

            dispersion * cov

        For (some subset of models) `dispersion` will typically be the mean
        square error from the estimated model (sigma^2)
        '''
        self.theta = theta
        self.Y = Y
        self.model = model
        if cov is None:
            self.cov = self.model.information(self.theta,
                                              nuisance=self.nuisance)
        else:
            self.cov = cov
        self.dispersion = dispersion
        self.nuisance = nuisance

        self.df_total = Y.shape[0]
        self.df_model = model.df_model
        # put this as a parameter of LikelihoodModel
        self.df_resid = self.df_total - self.df_model
        
    @setattr_on_read
    def logL(self):
        """
        The maximized log-likelihood
        """
        return self.model.logL(self.theta, self.Y, nuisance=self.nuisance)

    def t(self, column=None):
        """
        Return the (Wald) t-statistic for a given parameter estimate.

        Use Tcontrast for more complicated (Wald) t-statistics.
        """

        if column is None:
            column = range(self.theta.shape[0])

        column = np.asarray(column)
        _theta = self.theta[column]
        _cov = self.vcov(column=column)
        if _cov.ndim == 2:
            _cov = np.diag(_cov)
        _t = _theta * positive_reciprocal(np.sqrt(_cov))
        return _t

    def vcov(self, matrix=None, column=None, dispersion=None, other=None):
        """ Variance/covariance matrix of linear contrast

        Parameters
        ----------
        matrix: (dim, self.theta.shape[0]) array, optional
            numerical contrast specification, where ``dim`` refers to the
            'dimension' of the contrast i.e. 1 for t contrasts, 1 or more
            for F contrasts.

        column: int, optional
            alternative way of specifying contrasts (column index)

        dispersion: float or (n_voxels,) array, optional
            value(s) for the dispersion parameters

        other: (dim, self.theta.shape[0]) array, optional
            alternative contrast specification (?)

        Returns
        -------
        cov: (dim, dim) or (n_voxels, dim, dim) array
            the estimated covariance matrix/matrices

        Returns the variance/covariance matrix of a linear contrast of the
        estimates of theta, multiplied by `dispersion` which will often be an
        estimate of `dispersion`, like, sigma^2.

        The covariance of interest is either specified as a (set of) column(s)
        or a matrix.
        """
        if self.cov is None:
            raise ValueError('need covariance of parameters for computing' +\
                             '(unnormalized) covariances')

        if dispersion is None:
            dispersion = self.dispersion

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return self.cov[column, column] * dispersion
            else:
                return self.cov[column][:, column] * dispersion

        elif matrix is not None:
            if other is None:
                other = matrix
            tmp = np.dot(matrix, np.dot(self.cov, np.transpose(other)))
            if np.isscalar(dispersion):
                return tmp * dispersion
            else:
                return tmp[:, :, np.newaxis] * dispersion
        if matrix is None and column is None:
            return self.cov * dispersion

    def Tcontrast(self, matrix, store=('t', 'effect', 'sd'), dispersion=None):
        """ Compute a Tcontrast for a row vector `matrix`

        To get the t-statistic for a single column, use the 't' method.

        Parameters
        ----------
        matrix : 1D array-like
            contrast matrix

        store : sequence, optional
            components of t to store in results output object.  Defaults to all
            components ('t', 'effect', 'sd').

        dispersion : None or float, optional

        Returns
        -------
        res : ``TContrastResults`` object
        """
        matrix = np.asarray(matrix)
        # 1D vectors assumed to be row vector
        if matrix.ndim == 1:
            matrix = matrix[None]
        if matrix.shape[0] != 1:
            raise ValueError("t contrasts should have only one row")
        if matrix.shape[1] != self.theta.shape[0]:
            raise ValueError("t contrasts should be length P=%d, "
                             "but this is length %d" % (self.theta.shape[0],
                                                        matrix.shape[1]))
        store = set(store)
        if not store.issubset(('t', 'effect', 'sd')):
            raise ValueError('Unexpected store request in %s' % store)
        st_t = st_effect = st_sd = effect = sd = None
        if 't' in store or 'effect' in store:
            effect = np.dot(matrix, self.theta)
            if 'effect' in store:
                st_effect = np.squeeze(effect)
        if 't' in store or 'sd' in store:
            sd = np.sqrt(self.vcov(matrix=matrix, dispersion=dispersion))
            if 'sd' in store:
                st_sd = np.squeeze(sd)
        if 't' in store:
            st_t = np.squeeze(effect * positive_reciprocal(sd))
        return TContrastResults(effect=st_effect, t=st_t, sd=st_sd,
                                df_den=self.df_resid)

    def Fcontrast(self, matrix, dispersion=None, invcov=None):
        """ Compute an Fcontrast for a contrast matrix `matrix`.

        Here, `matrix` M is assumed to be non-singular. More precisely

        .. math::

            M pX pX' M'

        is assumed invertible. Here, :math:`pX` is the generalized inverse of
        the design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        See the contrast module to see how to specify contrasts.  In particular,
        the matrices from these contrasts will always be non-singular in the
        sense above.

        Parameters
        ----------
        matrix : 1D array-like
            contrast matrix

        dispersion : None or float, optional
            If None, use ``self.dispersion``

        invcov : None or array, optional
            Known inverse of variance covariance matrix.
            If None, calculate this matrix.

        Returns
        -------
        f_res : ``FContrastResults`` instance
            with attributes F, df_den, df_num

        Notes
        -----
        For F contrasts, we now specify an effect and covariance
        """
        matrix = np.asarray(matrix)
        # 1D vectors assumed to be row vector
        if matrix.ndim == 1:
            matrix = matrix[None]
        if matrix.shape[1] != self.theta.shape[0]:
            raise ValueError("F contrasts should have shape[1] P=%d, "
                             "but this has shape[1] %d" % (self.theta.shape[0],
                                                           matrix.shape[1]))
        ctheta = np.dot(matrix, self.theta)
        if matrix.ndim == 1:
            matrix = matrix.reshape((1, matrix.shape[0]))
        if dispersion is None:
            dispersion = self.dispersion
        q = matrix.shape[0]
        if invcov is None:
            invcov = inv(self.vcov(matrix=matrix, dispersion=1.0))
        F = np.add.reduce(np.dot(invcov, ctheta) * ctheta, 0) * \
            positive_reciprocal((q * dispersion))
        F = np.squeeze(F)
        return FContrastResults(
            effect=ctheta, covariance=self.vcov(
                matrix=matrix, dispersion=dispersion[np.newaxis]),
            F=F, df_den=self.df_resid, df_num=invcov.shape[0])

    def conf_int(self, alpha=.05, cols=None, dispersion=None):
        ''' The confidence interval of the specified theta estimates.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval.
            ie., `alpha` = .05 returns a 95% confidence interval.

        cols : tuple, optional
            `cols` specifies which confidence intervals to return

        dispersion : None or scalar
            scale factor for the variance / covariance (see class docstring and
            ``vcov`` method docstring)

        Returns
        -------
        cis : ndarray
            `cis` is shape ``(len(cols), 2)`` where each row contains [lower,
            upper] for the given entry in `cols`

        Examples
        --------
        >>> from numpy.random import standard_normal as stan
        >>> from nistats.regression import OLSModel
        >>> x = np.hstack((stan((30,1)),stan((30,1)),stan((30,1))))
        >>> beta=np.array([3.25, 1.5, 7.0])
        >>> y = np.dot(x,beta) + stan((30))
        >>> model = OLSModel(x).fit(y)
        >>> confidence_intervals = model.conf_int(cols=(1,2))

        Notes
        -----
        
        Confidence intervals are two-tailed.
        
        tails : string, optional
            Possible values: 'two' | 'upper' | 'lower'

        '''
        if cols is None:
            lower = self.theta - inv_t_cdf(1 - alpha / 2, self.df_resid) *\
                    np.sqrt(np.diag(self.vcov(dispersion=dispersion)))
            upper = self.theta + inv_t_cdf(1 - alpha / 2, self.df_resid) *\
                    np.sqrt(np.diag(self.vcov(dispersion=dispersion)))
        else:
            lower, upper = [], []
            for i in cols:
                lower.append(
                    self.theta[i] - inv_t_cdf(1 - alpha / 2, self.df_resid) *
                    np.sqrt(self.vcov(column=i, dispersion=dispersion)))
                upper.append(
                    self.theta[i] + inv_t_cdf(1 - alpha / 2, self.df_resid) *
                    np.sqrt(self.vcov(column=i, dispersion=dispersion)))
        return np.asarray(list(zip(lower, upper)))


class TContrastResults(object):
    """ Results from a t contrast of coefficients in a parametric model.

    The class does nothing, it is a container for the results from T contrasts,
    and returns the T-statistics when np.asarray is called.
    """

    def __init__(self, t, sd, effect, df_den=None):
        if df_den is None:
            df_den = np.inf
        self.t = t
        self.sd = sd
        self.effect = effect
        self.df_den = df_den

    def __array__(self):
        return np.asarray(self.t)

    def __str__(self):
        return ('<T contrast: effect=%s, sd=%s, t=%s, df_den=%d>' %
                (self.effect, self.sd, self.t, self.df_den))


class FContrastResults(object):
    """ Results from an F contrast of coefficients in a parametric model.

    The class does nothing, it is a container for the results from F contrasts,
    and returns the F-statistics when np.asarray is called.
    """

    def __init__(self, effect, covariance, F, df_num, df_den=None):
        if df_den is None:
            df_den = np.inf
        self.effect = effect
        self.covariance = covariance
        self.F = F
        self.df_den = df_den
        self.df_num = df_num
    def __array__(self):
        return np.asarray(self.F)

    def __str__(self):
        return '<F contrast: F=%s, df_den=%d, df_num=%d>' % \
            (repr(self.F), self.df_den, self.df_num)
