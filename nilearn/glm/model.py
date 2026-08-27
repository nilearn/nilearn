"""Implement classes to handle statistical tests on likelihood models."""

import numpy as np
from nibabel.onetime import auto_attr
from scipy.linalg import inv
from scipy.stats import t as t_distribution

from nilearn.glm._utils import pad_contrast, positive_reciprocal

# Inverse t cumulative distribution
inv_t_cdf = t_distribution.ppf


def _scaled_covariance_block(cov, matrix, column) -> np.ndarray | None:
    """Return the block of ``cov`` that :meth:`vcov` scales by dispersion.

    Returns None for the ``matrix`` branch, which the guard leaves
    alone: it appends an axis for a 1-D dispersion and behaves as it
    always has for every other shape. ``column`` is tested first because
    :meth:`vcov` runs its ``column`` branch whenever ``column`` is given,
    whether or not ``matrix`` is given with it.

    Indexes exactly as the branches of :meth:`vcov` do, so that a check
    made on the returned block cannot disagree with what is returned.

    Returns
    -------
    :obj:`numpy.ndarray` or None
        The block of ``cov`` the selection keeps, or None when the
        ``matrix`` branch will run.
    """
    if column is not None:
        column = np.asarray(column)
        if column.shape == ():
            return cov[column, column]
        return cov[column][:, column]
    if matrix is not None:
        return None
    return cov


def _uniform_covariance_stack(cov, matrix, column, dispersion) -> np.ndarray:
    """Return one covariance matrix per dispersion value.

    This is the shape :meth:`LikelihoodModelResults.vcov` returns
    unless ``uniform=False``: one copy of the selected block per
    dispersion value, whatever shape the dispersion itself arrives in.
    A scalar dispersion is one value, so it gives ``(1, dim, dim)``.
    The block is always square, so the result is always
    ``(n_dispersion, dim, dim)``. A ``matrix`` of rank 3 or more cannot
    give a square block, so it raises here rather than returning
    something of another rank.

    Selecting a single regressor gives a one by one block rather than
    dropping the axes, which is the difference from the branches of
    :meth:`LikelihoodModelResults.vcov`. Through ``matrix`` the block
    is the contrast's own dimension, so a single row spanning several
    regressors is also one by one.

    Returns
    -------
    :obj:`numpy.ndarray`
        The selected covariance block, scaled once per dispersion value.
    """
    if column is not None:
        column = np.asarray(column)
        selected = np.atleast_1d(np.arange(cov.shape[0])[column])
        if selected.ndim != 1:
            # A 2-D selector, or a 0-d boolean, which numpy reads as a
            # whole-array mask rather than as one regressor.
            raise ValueError(
                "column must be an integer, a 1-D sequence of integers "
                "or a 1-D boolean mask."
            )
        block = cov[selected][:, selected]
    elif matrix is not None:
        # Read the rank without rebinding matrix, so that an np.matrix
        # reaches np.dot as one and the asarray below still has work to
        # do.
        rank = np.asarray(matrix).ndim
        if rank > 2:
            # matrix is documented as (dim, n_regressors). A higher rank
            # one keeps its extra axes through the product, so the result
            # cannot be one square block per dispersion value. It used
            # to come back with those axes still on it, or fail inside
            # numpy, depending on both its shape and the number of
            # dispersion values.
            raise ValueError(
                "matrix must be 1-D or 2-D to give one covariance matrix "
                f"per dispersion value; this one has {rank} "
                "dimensions. Pass uniform=False for the older shapes."
            )
        block = np.dot(matrix, np.dot(cov, np.transpose(matrix)))
    else:
        block = cov
    # asarray because ``*`` means matrix multiplication for the
    # np.matrix a caller may have passed as ``matrix``.
    block = np.asarray(block)
    return block * np.ravel(dispersion)[:, np.newaxis, np.newaxis]


class LikelihoodModelResults:
    """Class to contain results from likelihood models.

    This is the class in which things like AIC, BIC, llf
    can be implemented as methods, not computed in, say,
    the fit method of OLSModel.

    Parameters
    ----------
    theta : ndarray
        Parameter estimates from estimated model.

    Y : ndarray
        Data.

    model : ``LikelihoodModel`` instance
        Model used to generate fit.

    cov : None or ndarray, default=None
        Covariance of thetas.

    dispersion : scalar, default=1
        Multiplicative factor in front of `cov`.

    nuisance : None of ndarray, default=None
        Parameter estimates needed to compute logL.

    Notes
    -----
    The covariance of thetas is given by:

        dispersion * cov

    For (some subset of models) `dispersion` will typically be the mean
    square error from the estimated model (sigma^2)

    """

    def __init__(
        self,
        theta,
        Y,
        model,
        cov=None,
        dispersion=1.0,
        nuisance=None,
    ):
        self.theta = theta
        self.Y = Y
        self.model = model
        if cov is None:
            self.cov = self.model.information(
                self.theta, nuisance=self.nuisance
            )
        else:
            self.cov = cov
        self.dispersion = dispersion
        self.nuisance = nuisance

        self.df_total = Y.shape[0]
        self.df_model = model.df_model
        # put this as a parameter of LikelihoodModel
        self.df_residuals = self.df_total - self.df_model

    # @auto_attr store the value as an object attribute after initial call
    # better performance than @property
    @auto_attr
    def logL(self):  # noqa: N802
        """Return the maximized log-likelihood."""
        return self.model.logL(self.theta, self.Y, nuisance=self.nuisance)

    def t(self, column=None):
        """
        Return the (Wald) t-statistic for a given parameter estimate.

        Use Tcontrast for more complicated (Wald) t-statistics.

        Parameters
        ----------
        column : :obj:`int`, sequence, or 1-D :obj:`bool` mask, \
default=None
            Which regressor(s) to return the t-statistic for. None
            means every regressor.

        Returns
        -------
        t : (n_regressors,) or (n_regressors, n_voxels) array
            One t-statistic per requested regressor, per column of the
            data the model was fitted on. A single integer ``column``
            returns the statistic for that regressor alone, without
            the regressor axis.

            .. nilearn_versionchanged:: 0.14.1
                For a model fitted on several columns of data, the
                per-column axis is kept instead of being dropped. Asking
                for several regressors at once used to raise, or to
                return wrong values when the number of columns of data
                happened to equal the number of regressors, or when the
                data was 2-D with a single column. A single regressor on
                several columns of data was also wrong, silently.

        """
        if column is None:
            column = range(self.theta.shape[0])

        column = np.asarray(column)
        if column.ndim > 1:
            # Iterating a 2-D selector would hand whole rows to vcov,
            # which selects blocks, and a block is not one regressor.
            raise ValueError(
                "column must be an integer, a 1-D sequence of integers "
                "or a 1-D boolean mask."
            )
        _theta = self.theta[column]
        if column.shape == ():
            _cov = self.vcov(column=column, uniform=False)
        else:
            # Ask for one regressor at a time. Asking for several at once
            # gives back a covariance matrix, whose diagonal has already
            # dropped the column-of-data axis that ``_theta`` still has.
            _cov = np.array(
                [
                    self.vcov(column=c, uniform=False)
                    for c in np.arange(self.theta.shape[0])[column]
                ]
            )
        _t = _theta * positive_reciprocal(np.sqrt(_cov))
        return _t

    def vcov(
        self,
        matrix=None,
        column=None,
        dispersion=None,
        *,
        uniform=True,
    ):
        """Return Variance/covariance matrix of linear :term:`contrast`.

        Parameters
        ----------
        matrix : (dim, self.theta.shape[0]) array, default=None
            Numerical :term:`contrast` specification,
            where ``dim`` refers to the 'dimension' of the contrast
            i.e. 1 for t contrasts, 1
            or more for F :term:`contrasts<contrast>`.

        column : :obj:`int`, sequence, or 1-D :obj:`bool` mask, \
default=None
            Alternative way of specifying :term:`contrasts<contrast>`
            (column index or indices). Takes precedence over ``matrix``
            when both are given.

        dispersion : :obj:`float` or (n_voxels,) array, default=None
            Value(s) for the dispersion parameters. One covariance
            matrix is returned per value. Under ``uniform=False``,
            several values on axes that overlap a covariance matrix,
            as in a 1-D array, a row or a column vector, broadcast
            along the matrix rather than stacking, so there they are
            only accepted where the result is not a matrix; reshape,
            for instance to ``dispersion[:, None, None]``, to get one
            matrix per value.

        uniform : :obj:`bool` or None, keyword only, default=True
            Whether to return one covariance matrix per dispersion
            value for every call. True gives ``(n_dispersion, dim,
            dim)``, where ``dim`` is the size of the block the
            selection keeps: the number of regressors ``column``
            names, or every regressor when neither ``column`` nor
            ``matrix`` is given, or the number of rows of ``matrix``
            when a contrast is given, whatever number of regressors
            that contrast spans. The block is always square, so the
            result is always ``(n_dispersion, dim, dim)``, and a
            ``matrix`` of rank 3 or more raises rather than returning
            something of another rank. False gives
            the older shapes listed below, which depend on the
            arguments; it warns from 0.15.0 and goes in 0.16.0. None
            means True.

            .. nilearn_versionadded:: 0.14.1

        Returns
        -------
        cov : array
            ``(n_dispersion, dim, dim)``, where ``n_dispersion`` is the
            number of dispersion values and ``dim`` is the size of the
            selected block, as described under ``uniform`` above.

            Under ``uniform=False`` the shape depends on the arguments:
            ``(dim, dim)`` for a single dispersion, ``(dim, dim,
            n_voxels)`` from ``matrix``, ``(n_voxels,)`` when ``column``
            selects a single regressor and there is one dispersion per
            column of data, ``(n_voxels, dim, dim)`` when the dispersion
            carries its own axes, as in ``dispersion[:, None, None]``,
            and a 0-d value when a single regressor meets a scalar
            dispersion.

            .. nilearn_versionchanged:: 0.14.1
                Every call that returns gives
                ``(n_dispersion, dim, dim)``, instead of a shape that
                depended on which arguments were given. A selector that
                cannot name one regressor at a time, a 2-D array or a
                0-d boolean, raises. ``uniform=False`` restores the
                older shapes, which is a one-line migration for code
                that read them. Under ``uniform=False``, asking for
                several regressors at once while the dispersion
                carries several values on axes overlapping the
                covariance matrix raises, instead of broadcasting the
                dispersions along the columns of the covariance
                matrix.

            .. nilearn_deprecated:: 0.14.1
                ``uniform=False``, and with it the argument-dependent
                shapes, will warn from 0.15.0 and be removed in
                0.16.0.

        Returns the variance/covariance matrix of a linear contrast of the
        estimates of theta, multiplied by `dispersion` which will often be an
        estimate of `dispersion`, like, sigma^2.

        The covariance of interest is either specified as a (set of) column(s)
        or a matrix.

        """
        if self.cov is None:
            raise ValueError(
                "need covariance of parameters for computing"
                "(unnormalized) covariances"
            )

        if dispersion is None:
            dispersion = self.dispersion

        # The commit that added this keyword documented None as meaning
        # False and warning. The warning is gone, so None now means the
        # default rather than the deprecated branch. The keyword has not
        # shipped, so this reverses a docstring rather than a release.
        if uniform is None:
            uniform = True

        # TODO (nilearn >= 0.15.0) warn on uniform=False, so the removal
        # is not the first notice anyone gets. All seven in-tree callers
        # ask for it and the suite runs with -W error::FutureWarning, so
        # they need a path that skips the warning before it can be added.
        # TODO (nilearn >= 0.16.0) drop the branch below and keep taking
        # the keyword, since 0.14.1 leaves uniform=True writable and
        # removing the argument would be a TypeError with no second
        # warning. uniform=False should raise there, not be ignored, or
        # the shape flips silently for whoever opted out. The callers
        # need rewriting rather than just losing the keyword: on the
        # uniform shape they broadcast into extra axes instead of
        # raising, so getting it wrong is silent.
        if uniform:
            return _uniform_covariance_stack(
                self.cov, matrix, column, dispersion
            )

        # One covariance matrix cannot hold one value per dispersion.
        # A dispersion whose values sit on axes overlapping the
        # covariance block (1-D, row and column vector alike) smears
        # the block; values on axes of the dispersion's own, as in
        # dispersion[:, None, None], stack whole blocks and pass, and
        # a single-regressor block is one number, so it always fits.
        block = _scaled_covariance_block(self.cov, matrix, column)
        # Order matters below: for a 0-d block, [-0:] would be the whole
        # dispersion shape, but a 0-d block always has size 1 and the
        # size check short-circuits first.
        smears = (
            block is not None
            and np.size(block) > 1
            and np.size(dispersion) > 1
            and any(s > 1 for s in np.shape(dispersion)[-np.ndim(block) :])
        )
        if smears:
            raise ValueError(
                "There is one covariance matrix per dispersion value "
                "here, so they cannot be returned as a single matrix. "
                "This is the older argument-dependent behavior, which "
                "you asked for with uniform=False: drop it to get one "
                "matrix per dispersion value. To stay on the older "
                "shapes, select a single regressor at a time, or pass "
                "a scalar dispersion, or reshape the dispersion to "
                "carry its own axes, as in dispersion[:, None, None], "
                "or pass matrix= with a 1-D dispersion and no column, "
                "since column takes precedence over matrix."
            )

        if matrix is None and column is None:
            return self.cov * dispersion

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return self.cov[column, column] * dispersion
            else:
                return self.cov[column][:, column] * dispersion

        else:
            tmp = np.dot(matrix, np.dot(self.cov, np.transpose(matrix)))
            if np.isscalar(dispersion):
                return tmp * dispersion
            else:
                return tmp[:, :, np.newaxis] * dispersion

    def Tcontrast(self, matrix, store=("t", "effect", "sd"), dispersion=None):  # noqa: N802
        """Compute a Tcontrast for a row vector `matrix`.

        To get the t-statistic for a single column, use the 't' method.

        Parameters
        ----------
        matrix : 1D array-like
            Contrast matrix.

        store : sequence, default=('t', 'effect', 'sd')
            Components of t to store in results output object.

        dispersion : None or :obj:`float`, default = None

        Returns
        -------
        res : ``TContrastResults`` object

        """
        matrix = np.asarray(matrix)
        # 1D vectors assumed to be row vector
        if matrix.ndim == 1:
            matrix = matrix[None]
        if matrix.size == 0:
            raise ValueError(f"t contrasts cannot be empty: got {matrix}")
        if matrix.shape[0] != 1:
            raise ValueError(
                f"t contrasts should have only one row: got {matrix}."
            )
        matrix = pad_contrast(con_val=matrix, theta=self.theta, stat_type="t")
        store = set(store)
        if not store.issubset(("t", "effect", "sd")):
            raise ValueError(f"Unexpected store request in {store}")
        st_t = st_effect = st_sd = effect = sd = None
        if "t" in store or "effect" in store:
            effect = np.dot(matrix, self.theta)
        if "effect" in store:
            st_effect = np.squeeze(effect)
        if "t" in store or "sd" in store:
            sd = np.sqrt(
                self.vcov(matrix=matrix, dispersion=dispersion, uniform=False)
            )
        if "sd" in store:
            st_sd = np.squeeze(sd)
        if "t" in store:
            st_t = np.squeeze(effect * positive_reciprocal(sd))
        return TContrastResults(
            effect=st_effect, t=st_t, sd=st_sd, df_den=self.df_residuals
        )

    def Fcontrast(self, matrix, dispersion=None, invcov=None):  # noqa: N802
        """Compute an F contrast for a :term:`contrast` matrix ``matrix``.

        Here, ``matrix`` M is assumed to be non-singular. More precisely

        .. math::

            M pX pX' M'

        is assumed invertible. Here, :math:`pX` is the generalized inverse of
        the design matrix of the model.
        There can be problems in non-OLS models where
        the rank of the covariance of the noise is not full.

        See the contrasts module to see how to specify contrasts.
        In particular, the matrices from these contrasts will always be
        non-singular in the sense above.

        Parameters
        ----------
        matrix : 1D array-like
            Contrast matrix.

        dispersion : None or :obj:`float`, default=None
            If None, use ``self.dispersion``.

        invcov : None or array, default=None
            Known inverse of variance covariance matrix.
            If None, calculate this matrix.

        Returns
        -------
        f_res : ``FContrastResults`` instance
            with attributes F, df_den, df_num

        Notes
        -----
        For F contrasts, we now specify an effect and covariance.

        """
        matrix = np.asarray(matrix)
        # 1D vectors assumed to be row vector
        if matrix.ndim == 1:
            matrix = matrix[None]
        if matrix.shape[1] != self.theta.shape[0]:
            raise ValueError(
                f"F contrasts should have shape[1]={self.theta.shape[0]}, "
                f"but this has shape[1]={matrix.shape[1]}"
            )
        matrix = pad_contrast(con_val=matrix, theta=self.theta, stat_type="F")
        ctheta = np.dot(matrix, self.theta)
        if matrix.ndim == 1:
            matrix = matrix.reshape((1, matrix.shape[0]))
        if dispersion is None:
            dispersion = self.dispersion
        q = matrix.shape[0]
        if invcov is None:
            invcov = inv(
                self.vcov(matrix=matrix, dispersion=1.0, uniform=False)
            )
        F = np.add.reduce(
            np.dot(invcov, ctheta) * ctheta, 0
        ) * positive_reciprocal(q * dispersion)
        F = np.squeeze(F)
        return FContrastResults(
            effect=ctheta,
            covariance=self.vcov(
                matrix=matrix,
                dispersion=dispersion[np.newaxis],
                uniform=False,
            ),
            F=F,
            df_den=self.df_residuals,
            df_num=invcov.shape[0],
        )

    def conf_int(self, alpha=0.05, cols=None, dispersion=None):
        """Return the confidence interval of the specified theta estimates.

        Parameters
        ----------
        alpha : :obj:`float`, default=0.05
            The `alpha` level for the confidence interval.
            ie., `alpha` = .05 returns a 95% confidence interval.


        cols : sequence of :obj:`int` or 1-D :obj:`bool` mask, \
default=None
            `cols` specifies which confidence intervals to return. A
            boolean mask selects regressors the way the ``column``
            argument of :meth:`t` does. A bare integer is not
            accepted, as before.

        dispersion : None, scalar or array-like, default=None
            Scale factor for the variance / covariance
            (see class docstring and ``vcov`` method docstring).

        Returns
        -------
        cis : ndarray
            `cis` is shape ``(len(cols), 2)``, or ``(len(cols), 2,
            n_voxels)`` when the model was fitted on several columns of
            data, where each row contains [lower, upper] for the given
            entry in `cols`

            .. nilearn_versionchanged:: 0.14.1
                For a model fitted on several columns of data, the
                per-column axis is kept instead of being dropped. A
                1-D boolean mask in ``cols`` now selects regressors, as
                it does in :meth:`t`, instead of being iterated as 0-d
                boolean indices. An explicit non-scalar dispersion
                keeps its own axis instead of being collapsed through
                ``np.diag``, including a size-1 array.

        Notes
        -----
        Confidence intervals are two-tailed.

        Examples
        --------
        >>> from numpy.random import standard_normal as stan
        >>> from nilearn.glm import OLSModel
        >>> x = np.hstack((stan((30, 1)), stan((30, 1)), stan((30, 1))))
        >>> beta = np.array([3.25, 1.5, 7.0])
        >>> y = np.dot(x, beta) + stan((30))
        >>> model = OLSModel(x).fit(y)
        >>> confidence_intervals = model.conf_int(cols=(1, 2))

        """
        if cols is None:
            # Take the regressors one at a time, as the explicit branch
            # below does. Asking for all of them at once goes through a
            # covariance matrix, which cannot hold one variance per
            # column of data.
            cols = range(self.theta.shape[0])
        else:
            # Materialize any iterable, so generators and sets keep
            # working exactly as they always have.
            if not isinstance(cols, np.ndarray):
                cols = list(cols)
            arr = np.asarray(cols)
            if arr.ndim > 1:
                # Iterating a 2-D selector would hand whole rows to
                # vcov, which selects blocks, not single regressors.
                raise ValueError(
                    "cols must be a 1-D sequence of integers or a 1-D "
                    "boolean mask."
                )
            if arr.dtype == np.bool_ and arr.ndim == 1:
                # A 1-D boolean mask selects regressors, as it does in
                # t(). Iterating it raw would hand 0-d booleans to
                # vcov, and a 0-d boolean indexes a block, not an
                # element.
                cols = np.arange(self.theta.shape[0])[arr]
        if dispersion is not None and not np.isscalar(dispersion):
            # vcov scales one covariance element, a numpy scalar, by
            # the dispersion, and a numpy scalar times a python list is
            # a TypeError rather than an array.
            dispersion = np.asarray(dispersion)
        lower, upper = [], []
        for i in cols:
            half_width = inv_t_cdf(1 - alpha / 2, self.df_residuals) * np.sqrt(
                self.vcov(column=i, dispersion=dispersion, uniform=False)
            )
            lower.append(self.theta[i] - half_width)
            upper.append(self.theta[i] + half_width)
        return np.asarray(list(zip(lower, upper, strict=False)))


class TContrastResults:
    """Results from a t :term:`contrast` of coefficients in a parametric model.

    The class does nothing.
    It is a container for the results from T :term:`contrasts<contrast>`,
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
        return (
            "<T contrast: "
            f"effect={self.effect}, "
            f"sd={self.sd}, "
            f"t={self.t}, "
            f"df_den={self.df_den}>"
        )


class FContrastResults:
    """Results from an F :term:`contrast` of coefficients \
       in a parametric model.

    The class does nothing.
    It is a container for the results from F :term:`contrasts<contrast>`,
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
        return (
            "<F contrast: "
            f"F={self.F!r}, "
            f"df_den={self.df_den}, "
            f"df_num={self.df_num}>"
        )
