"""Implement some standard regression models: OLS and WLS \
models, as well as an AR(p) regression model.

Models are specified with a design matrix and are fit using their
'fit' method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

'Introduction to Linear Regression Analysis', Douglas C. Montgomery,
    Elizabeth A. Peck, G. Geoffrey Vining. Wiley, 2006.

"""

__docformat__ = "restructuredtext en"

import numpy as np
import scipy.linalg as spl
from nibabel.onetime import auto_attr
from numpy.linalg import matrix_rank

from nilearn.glm._utils import positive_reciprocal
from nilearn.glm.model import LikelihoodModelResults


class OLSModel:
    """A simple ordinary least squares model.

    Parameters
    ----------
    design : array-like
        This is your design matrix.  Data are assumed to be column ordered
        with observations in rows.

    Methods
    -------
    model.__init___(design)
    model.logL(b=self.beta, Y)

    Attributes
    ----------
    design : ndarray
        This is the design, or X, matrix.

    whitened_design : ndarray
        This is the whitened design matrix.
        `design` == `whitened_design` by default for the OLSModel,
        though models that inherit from the OLSModel will whiten the design.

    calc_beta : ndarray
        This is the Moore-Penrose pseudoinverse of the whitened design matrix.

    normalized_cov_beta : ndarray
        ``np.dot(calc_beta, calc_beta.T)``

    df_residuals : scalar
        Degrees of freedom of the residuals.  Number of observations less the
        rank of the design.

    df_model : scalar
        Degrees of freedome of the model.  The rank of the design.

    """

    def __init__(self, design):
        super().__init__()
        self.initialize(design)

    def initialize(self, design):
        """Construct instance."""
        # PLEASE don't assume we have a constant...
        # TODO: handle case for noconstant regression
        self.design = design
        self.whitened_design = self.whiten(self.design)
        self.calc_beta = spl.pinv(self.whitened_design)
        self.normalized_cov_beta = np.dot(
            self.calc_beta, np.transpose(self.calc_beta)
        )
        self.df_total = self.whitened_design.shape[0]

        eps = np.abs(self.design).sum() * np.finfo(np.float64).eps
        self.df_model = matrix_rank(self.design, eps)
        self.df_residuals = self.df_total - self.df_model

    def logL(self, beta, Y, nuisance=None):  # noqa: N802
        r"""Return the value of the loglikelihood function at beta.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector, :term:`beta<Beta>`,
        for the dependent variable, Y
        and the nuisance parameter, sigma :footcite:t:`Greene2003`.

        Parameters
        ----------
        beta : ndarray
            The parameter estimates.  Must be of length ``df_model``.

        Y : ndarray
            The dependent variable

        nuisance : dict, optional
            A dict with key 'sigma', which is an optional estimate of sigma.
            If None, defaults to its maximum likelihood estimate
            (with beta fixed) as
            ``sum((Y - X*beta)**2) / n``, where n=Y.shape[0], X=self.design.

        Returns
        -------
        loglf : float
            The value of the loglikelihood function.

        Notes
        -----
        The log-Likelihood Function is defined as

        .. math::

            \ell(\beta,\sigma,Y)=
            -\frac{n}{2}\log(2\pi\sigma^2) - \|Y-X\beta\|^2/(2\sigma^2)

        The parameter :math:`\sigma` above is what is sometimes referred to
        as a nuisance parameter. That is, the likelihood is considered as a
        function of :math:`\beta`, but to evaluate it, a value of
        :math:`\sigma` is needed.

        If :math:`\sigma` is not provided,
        then its maximum likelihood estimate:

        .. math::

            \hat{\sigma}(\beta) = \frac{\text{SSE}(\beta)}{n}

        is plugged in. This likelihood is now a function of only :math:`\beta`
        and is technically referred to as a profile-likelihood.

        References
        ----------
        .. footbibliography::

        """
        # This is overwriting an abstract method of LikelihoodModel
        X = self.whitened_design
        wY = self.whiten(Y)
        r = wY - np.dot(X, beta)
        n = self.df_total
        SSE = (r**2).sum(0)
        sigmasq = SSE / n if nuisance is None else nuisance["sigma"]

        loglf = -n / 2.0 * np.log(2 * np.pi * sigmasq) - SSE / (2 * sigmasq)
        return loglf

    def whiten(self, X):
        """Whiten design matrix.

        Parameters
        ----------
        X : array
            design matrix

        Returns
        -------
        whitened_X : array
            This matrix is the matrix whose pseudoinverse is ultimately
            used in estimating the coefficients. For OLSModel, it is
            does nothing. For WLSmodel, ARmodel, it pre-applies
            a square root of the covariance matrix to X.

        """
        return X

    def fit(self, Y):
        """Fit model to data `Y`.

        Full fit of the model including estimate of covariance matrix,
        (whitened) residuals and scale.

        Parameters
        ----------
        Y : array-like
            The dependent variable for the Least Squares problem.

        Returns
        -------
        fit : RegressionResults

        """
        # Other estimates of the covariance matrix for a heteroscedastic
        # regression model can be implemented in WLSmodel. (Weighted least
        # squares models assume covariance is diagonal, i.e. heteroscedastic).
        wY = self.whiten(Y)
        beta = np.dot(self.calc_beta, wY)
        wresid = wY - np.dot(self.whitened_design, beta)
        dispersion = np.sum(wresid**2, 0) / (
            self.whitened_design.shape[0] - self.whitened_design.shape[1]
        )
        lfit = RegressionResults(
            beta,
            Y,
            self,
            wY,
            wresid,
            dispersion=dispersion,
            cov=self.normalized_cov_beta,
        )
        return lfit


class ARModel(OLSModel):
    """A regression model with an AR(p) covariance structure.

    In terms of a LikelihoodModel, the parameters
    are beta, the usual regression parameters,
    and sigma, a scalar nuisance parameter that
    shows up as multiplier in front of the AR(p) covariance.

    Parameters
    ----------
    design : ndarray
        2D array with design matrix.

    rho : int or array-like
        If int, gives order of model, and initializes rho to zeros.  If
        ndarray, gives initial estimate of rho. Be careful as ``ARModel(X,
        1) != ARModel(X, 1.0)``.

    """

    def __init__(self, design, rho):
        if isinstance(rho, int):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0, 1]:
                raise ValueError("AR parameters must be a scalar or a vector")
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        super().__init__(design)

    def whiten(self, X):
        """Whiten a series of columns according to AR(p) covariance structure.

        Parameters
        ----------
        X : array-like of shape (n_features)
            Array to whiten.

        Returns
        -------
        whitened_X : ndarray
            X whitened with order self.order AR.

        """
        X = np.asarray(X, np.float64)
        whitened_X = X.copy()
        for i in range(self.order):
            whitened_X[(i + 1) :] = (
                whitened_X[(i + 1) :] - self.rho[i] * X[: -(i + 1)]
            )
        return whitened_X


class RegressionResults(LikelihoodModelResults):
    """Summarize the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.

    """

    def __init__(
        self,
        theta,
        Y,
        model,
        whitened_Y,
        whitened_residuals,
        cov=None,
        dispersion=1.0,
        nuisance=None,
    ):
        """See LikelihoodModelResults constructor.

        The only difference is that the whitened Y and residual values
        are stored for a regression model.

        """
        LikelihoodModelResults.__init__(
            self, theta, Y, model, cov, dispersion, nuisance
        )
        self.whitened_Y = whitened_Y
        self.whitened_residuals = whitened_residuals
        self.whitened_design = model.whitened_design

    # @auto_attr store the value as an object attribute after initial call
    # better performance than @property
    @auto_attr
    def residuals(self):
        """Residuals from the fit."""
        return self.Y - self.predicted

    @auto_attr
    def normalized_residuals(self):
        """Residuals, normalized to have unit length.

        See :footcite:t:`Montgomery2006` and :footcite:t:`Davidson2004`.

        Notes
        -----
        Is this supposed to return "standardized residuals,"
        residuals standardized
        to have mean zero and approximately unit variance?

        d_i = e_i / sqrt(MS_E)

        Where MS_E = SSE / (n - k)

        References
        ----------
        .. footbibliography::

        """
        return self.residuals * positive_reciprocal(np.sqrt(self.dispersion))

    @auto_attr
    def predicted(self):
        """Return linear predictor values from a design matrix."""
        beta = self.theta
        # the LikelihoodModelResults has parameters named 'theta'
        X = self.whitened_design
        return np.dot(X, beta)

    @auto_attr
    def SSE(self):  # noqa: N802
        """Error sum of squares.

        If not from an OLS model this is "pseudo"-SSE.
        """
        return (self.whitened_residuals**2).sum(0)

    @auto_attr
    def r_square(self):
        """Proportion of explained variance.

        If not from an OLS model this is "pseudo"-R2.
        """
        return np.var(self.predicted, 0) / np.var(self.whitened_Y, 0)

    @auto_attr
    def MSE(self):  # noqa: N802
        """Return Mean square (error)."""
        return self.SSE / self.df_residuals


class SimpleRegressionResults(LikelihoodModelResults):
    """Contain only information of the model fit necessary \
    for :term:`contrast` computation.

    Its intended to save memory when details of the model are unnecessary.

    """

    def __init__(self, results):
        """See LikelihoodModelResults constructor.

        The only difference is that the whitened Y and residual values
        are stored for a regression model.
        """
        self.theta = results.theta
        self.cov = results.cov
        self.dispersion = results.dispersion
        self.nuisance = results.nuisance

        self.df_total = results.Y.shape[0]
        self.df_model = results.model.df_model
        # put this as a parameter of LikelihoodModel
        self.df_residuals = self.df_total - self.df_model

    def logL(self):  # noqa: N802
        """Return the maximized log-likelihood."""
        raise NotImplementedError(
            "logL not implemented for "
            "SimpleRegressionsResults. "
            "Use RegressionResults"
        )

    def residuals(self, Y, X):
        """Residuals from the fit."""
        return Y - self.predicted(X)

    def normalized_residuals(self, Y, X):
        """Residuals, normalized to have unit length.

        See :footcite:t:`Montgomery2006` and :footcite:t:`Davidson2004`.

        Notes
        -----
        Is this supposed to return "standardized residuals,"
        residuals standardized
        to have mean zero and approximately unit variance?

        d_i = e_i / sqrt(MS_E)

        Where MS_E = SSE / (n - k)

        References
        ----------
        .. footbibliography::

        """
        return self.residuals(Y, X) * positive_reciprocal(
            np.sqrt(self.dispersion)
        )

    def predicted(self, X):
        """Return linear predictor values from a design matrix."""
        beta = self.theta
        return np.dot(X, beta)
