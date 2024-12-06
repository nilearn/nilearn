"""Contrast computation and operation on contrast to \
obtain fixed effect results.

Author: Bertrand Thirion, Martin Perez-Guevara, Ana Luisa Pinho 2020
"""

from warnings import warn

import numpy as np
import pandas as pd
import scipy.stats as sps

from nilearn._utils import logger, rename_parameters
from nilearn.glm._utils import pad_contrast, z_score
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.surface import SurfaceImage

DEF_TINY = 1e-50
DEF_DOFMAX = 1e10


def expression_to_contrast_vector(expression, design_columns):
    """Convert a string describing a :term:`contrast` \
       to a :term:`contrast` vector.

    Parameters
    ----------
    expression : :obj:`str`
        The expression to convert to a vector.

    design_columns : :obj:`list` or array of strings
        The column names of the design matrix.

    """
    if expression in design_columns:
        contrast_vector = np.zeros(len(design_columns))
        contrast_vector[list(design_columns).index(expression)] = 1.0
        return contrast_vector

    eye_design = pd.DataFrame(
        np.eye(len(design_columns)), columns=design_columns
    )
    try:
        contrast_vector = eye_design.eval(
            expression, engine="python"
        ).to_numpy()
    except Exception:
        raise ValueError(
            f"The expression ({expression}) is not valid. "
            "This could be due to "
            "defining the contrasts using design matrix columns that are "
            "invalid python identifiers."
        )

    return contrast_vector


@rename_parameters(
    replacement_params={"contrast_type": "stat_type"}, end_version="0.13.0"
)
def compute_contrast(labels, regression_result, con_val, stat_type=None):
    """Compute the specified :term:`contrast` given an estimated glm.

    Parameters
    ----------
    labels : array of shape (n_voxels,)
        A map of values on voxels used to identify the corresponding model

    regression_result : :obj:`dict`
        With keys corresponding to the different labels
        values are RegressionResults instances corresponding to the voxels.

    con_val : numpy.ndarray of shape (p) or (q, p)
        Where q = number of :term:`contrast` vectors
        and p = number of regressors.

    stat_type : {None, 't', 'F'}, default=None
        Type of the :term:`contrast`.
        If None, then defaults to 't' for 1D `con_val`
        and 'F' for 2D `con_val`.

    contrast_type :

        .. deprecated:: 0.10.3

            Use ``stat_type`` instead (see above).

    Returns
    -------
    con : Contrast instance,
        Yields the statistics of the :term:`contrast`
        (:term:`effects<Parameter Estimate>`, variance, p-values).

    """
    con_val = np.asarray(con_val)
    dim = 1
    if con_val.ndim > 1:
        dim = con_val.shape[0]

    if stat_type is None:
        stat_type = "t" if dim == 1 else "F"

    acceptable_stat_types = ["t", "F"]
    if stat_type not in acceptable_stat_types:
        raise ValueError(
            f"'{stat_type}' is not a known contrast type. "
            f"Allowed types are {acceptable_stat_types}."
        )

    if stat_type == "t":
        effect_ = np.zeros(labels.size)
        var_ = np.zeros(labels.size)
        for label_ in regression_result:
            label_mask = labels == label_
            reg = regression_result[label_].Tcontrast(con_val)
            effect_[label_mask] = reg.effect.T
            var_[label_mask] = (reg.sd**2).T

    elif stat_type == "F":
        from scipy.linalg import sqrtm

        effect_ = np.zeros((dim, labels.size))
        var_ = np.zeros(labels.size)
        # TODO
        # explain why we cannot simply do
        # reg = regression_result[label_].Tcontrast(con_val)
        # like above or refactor the code so it can be done
        for label_ in regression_result:
            label_mask = labels == label_
            reg = regression_result[label_]
            con_val = pad_contrast(
                con_val=con_val, theta=reg.theta, stat_type=stat_type
            )
            cbeta = np.atleast_2d(np.dot(con_val, reg.theta))
            invcov = np.linalg.inv(
                np.atleast_2d(reg.vcov(matrix=con_val, dispersion=1.0))
            )
            wcbeta = np.dot(sqrtm(invcov), cbeta)
            rss = reg.dispersion
            effect_[:, label_mask] = wcbeta
            var_[label_mask] = rss

    dof_ = regression_result[label_].df_residuals
    return Contrast(
        effect=effect_,
        variance=var_,
        dim=dim,
        dof=dof_,
        stat_type=stat_type,
    )


def compute_fixed_effect_contrast(labels, results, con_vals, stat_type=None):
    """Compute the summary contrast assuming fixed effects.

    Adds the same contrast applied to all labels and results lists.

    """
    contrast = None
    n_contrasts = 0
    for i, (lab, res, con_val) in enumerate(zip(labels, results, con_vals)):
        if np.all(con_val == 0):
            warn(f"Contrast for run {int(i)} is null.")
            continue
        contrast_ = compute_contrast(lab, res, con_val, stat_type)
        contrast = contrast_ if contrast is None else contrast + contrast_
        n_contrasts += 1
    if contrast is None:
        raise ValueError("All contrasts provided were null contrasts.")
    return contrast * (1.0 / n_contrasts)


class Contrast:
    """The contrast class handles the estimation \
    of statistical :term:`contrasts<contrast>` \
    on a given model: student (t) or Fisher (F).

    The important feature is that it supports addition,
    thus opening the possibility of fixed-effects models.

    The current implementation is meant to be simple,
    and could be enhanced in the future on the computational side
    (high-dimensional F :term:`contrasts<contrast>`
    may lead to memory breakage).

    Parameters
    ----------
    effect : array of shape (contrast_dim, n_voxels)
        The effects related to the :term:`contrast`.

    variance : array of shape (n_voxels)
        The associated variance estimate.

    dim : :obj:`int` or None, optional
        The dimension of the :term:`contrast`.

    dof : scalar, default=DEF_DOFMAX
        The degrees of freedom of the residuals.

    stat_type : {'t', 'F'}, default='t'
        Specification of the :term:`contrast` type.

    contrast_type :

        .. deprecated:: 0.10.3

            Use ``stat_type`` instead (see above).

    tiny : :obj:`float`, default=DEF_TINY
        Small quantity used to avoid numerical underflows.

    dofmax : scalar, default=DEF_DOFMAX
        The maximum degrees of freedom of the residuals.
    """

    @rename_parameters(
        replacement_params={"contrast_type": "stat_type"}, end_version="0.13.0"
    )
    def __init__(
        self,
        effect,
        variance,
        dim=None,
        dof=DEF_DOFMAX,
        stat_type="t",
        tiny=DEF_TINY,
        dofmax=DEF_DOFMAX,
    ):
        if variance.ndim != 1:
            raise ValueError("Variance array should have 1 dimension")
        if effect.ndim > 2:
            raise ValueError("Effect array should have 1 or 2 dimensions")

        self.effect = effect
        self.variance = variance
        self.dof = float(dof)
        if dim is None:
            self.dim = effect.shape[0] if effect.ndim == 2 else 1
        else:
            self.dim = dim

        if self.dim > 1 and stat_type == "t":
            logger.log(
                "Automatically converted multi-dimensional t to F contrast"
            )
            stat_type = "F"
        if stat_type not in ["t", "F"]:
            raise ValueError(
                f"{stat_type} is not a valid stat_type. Should be t or F"
            )
        self.stat_type = stat_type
        self.stat_ = None
        self.p_value_ = None
        self.one_minus_pvalue_ = None
        self.baseline = 0
        self.tiny = tiny
        self.dofmax = dofmax

    @property
    def contrast_type(self):
        """Return value of stat_type.

        .. deprecated:: 0.10.3
        """
        attrib_deprecation_msg = (
            'The attribute "contrast_type" '
            "will be removed in 0.13.0 release of Nilearn. "
            'Please use the attribute "stat_type" instead.'
        )
        warn(
            category=DeprecationWarning,
            message=attrib_deprecation_msg,
            stacklevel=2,
        )
        return self.stat_type

    def effect_size(self):
        """Make access to summary statistics more straightforward \
        when computing contrasts.
        """
        return self.effect

    def effect_variance(self):
        """Make access to summary statistics more straightforward \
        when computing contrasts.
        """
        return self.variance

    def stat(self, baseline=0.0):
        """Return the decision statistic associated with the test of the \
        null hypothesis: (H0) 'contrast equals baseline'.

        Parameters
        ----------
        baseline : :obj:`float`, default=0.0
            Baseline value for the test statistic.

        Returns
        -------
        stat : 1-d array, shape=(n_voxels,)
            statistical values, one per voxel.

        """
        self.baseline = baseline

        # Case: one-dimensional contrast ==> t or t**2
        if self.stat_type == "F":
            stat = (
                np.sum((self.effect - baseline) ** 2, 0)
                / self.dim
                / np.maximum(self.variance, self.tiny)
            )
        elif self.stat_type == "t":
            # avoids division by zero
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, self.tiny)
            )
        else:
            raise ValueError("Unknown statistic type")
        self.stat_ = stat.ravel()
        return self.stat_

    def p_value(self, baseline=0.0):
        """Return a parametric estimate of the p-value associated with \
        the null hypothesis (H0): 'contrast equals baseline', \
        using the survival function.

        Parameters
        ----------
        baseline : :obj:`float`, default=0.0
            Baseline value for the test statistic.


        Returns
        -------
        p_values : 1-d array, shape=(n_voxels,)
            p-values, one per voxel

        """
        if self.stat_ is None or self.baseline != baseline:
            self.stat_ = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.stat_type == "t":
            p_values = sps.t.sf(self.stat_, np.minimum(self.dof, self.dofmax))
        elif self.stat_type == "F":
            p_values = sps.f.sf(
                self.stat_, self.dim, np.minimum(self.dof, self.dofmax)
            )
        else:
            raise ValueError("Unknown statistic type")
        self.p_value_ = p_values
        return p_values

    def one_minus_pvalue(self, baseline=0.0):
        """Return a parametric estimate of the 1 - p-value associated \
        with the null hypothesis (H0): 'contrast equals baseline', \
        using the cumulative distribution function, \
        to ensure numerical stability.

        Parameters
        ----------
        baseline : :obj:`float`, default=0.0
            Baseline value for the test statistic.


        Returns
        -------
        one_minus_pvalues : 1-d array, shape=(n_voxels,)
            one_minus_pvalues, one per voxel

        """
        if self.stat_ is None or self.baseline != baseline:
            self.stat_ = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.stat_type == "t":
            one_minus_pvalues = sps.t.cdf(
                self.stat_, np.minimum(self.dof, self.dofmax)
            )
        else:
            assert self.stat_type == "F"
            one_minus_pvalues = sps.f.cdf(
                self.stat_, self.dim, np.minimum(self.dof, self.dofmax)
            )
        self.one_minus_pvalue_ = one_minus_pvalues
        return one_minus_pvalues

    def z_score(self, baseline=0.0):
        """Return a parametric estimation of the z-score associated \
        with the null hypothesis: (H0) 'contrast equals baseline'.

        Parameters
        ----------
        baseline : :obj:`float`, optional, default=0.0
            Baseline value for the test statistic.


        Returns
        -------
        z_score : 1-d array, shape=(n_voxels,)
            statistical values, one per voxel

        """
        if self.p_value_ is None or self.baseline != baseline:
            self.p_value_ = self.p_value(baseline)
        if self.one_minus_pvalue_ is None:
            self.one_minus_pvalue_ = self.one_minus_pvalue(baseline)

        # Avoid inf values kindly supplied by scipy.
        self.z_score_ = z_score(
            self.p_value_, one_minus_pvalue=self.one_minus_pvalue_
        )
        return self.z_score_

    def __add__(self, other):
        """Add two contrast, Yields an new Contrast instance.

        This should be used only on indepndent contrasts.
        """
        if self.stat_type != other.stat_type:
            raise ValueError(
                "The two contrasts do not have consistent type dimensions"
            )
        if self.dim != other.dim:
            raise ValueError(
                "The two contrasts do not have compatible dimensions"
            )
        dof_ = self.dof + other.dof
        if self.stat_type == "F":
            warn(
                "Running approximate fixed effects on F statistics.",
                category=UserWarning,
                stacklevel=2,
            )
        effect_ = self.effect + other.effect
        variance_ = self.variance + other.variance
        return Contrast(
            effect=effect_,
            variance=variance_,
            dim=self.dim,
            dof=dof_,
            stat_type=self.stat_type,
        )

    def __rmul__(self, scalar):
        """Multiply a contrast by a scalar."""
        scalar = float(scalar)
        effect_ = self.effect * scalar
        variance_ = self.variance * scalar**2
        dof_ = self.dof
        return Contrast(
            effect=effect_,
            variance=variance_,
            dof=dof_,
            stat_type=self.stat_type,
        )

    __mul__ = __rmul__

    def __div__(self, scalar):
        return self.__rmul__(1 / float(scalar))


def compute_fixed_effects(
    contrast_imgs,
    variance_imgs,
    mask=None,
    precision_weighted=False,
    dofs=None,
    return_z_score=False,
):
    """Compute the fixed effects, given images of effects and variance.

    Parameters
    ----------
    contrast_imgs : :obj:`list` of Nifti1Images or :obj:`str`\
                    or :obj:`~nilearn.surface.SurfaceImage`
        The input contrast images.

    variance_imgs : :obj:`list` of Nifti1Images or :obj:`str` \
                    or :obj:`~nilearn.surface.SurfaceImage`
        The input variance images.

    mask : Nifti1Image or NiftiMasker instance or \
        :obj:`~nilearn.maskers.SurfaceMasker` instance
        or None, default=None
        Mask image. If ``None``, it is recomputed from ``contrast_imgs``.

    precision_weighted : :obj:`bool`, default=False
        Whether fixed effects estimates should be weighted by inverse
        variance or not.

    dofs : array-like or None, default=None
        the degrees of freedom of the models with ``len = len(variance_imgs)``
        when ``None``,
        it is assumed that the degrees of freedom are 100 per input.

    return_z_score : :obj:`bool`, default=False
        Whether ``fixed_fx_z_score_img`` should be output or not.

    Returns
    -------
    fixed_fx_contrast_img : Nifti1Image or :obj:`~nilearn.surface.SurfaceImage`
        The fixed effects contrast computed within the mask.

    fixed_fx_variance_img : Nifti1Image or :obj:`~nilearn.surface.SurfaceImage`
        The fixed effects variance computed within the mask.

    fixed_fx_stat_img : Nifti1Image or :obj:`~nilearn.surface.SurfaceImage`
        The fixed effects stat computed within the mask.

    fixed_fx_z_score_img : Nifti1Image, optional
        The fixed effects corresponding z-transform

    Warns
    -----
    DeprecationWarning
        Starting in version 0.13, fixed_fx_z_score_img will always be returned

    """
    n_runs = len(contrast_imgs)
    if n_runs != len(variance_imgs):
        raise ValueError(
            f"The number of contrast images ({len(contrast_imgs)}) differs "
            f"from the number of variance images ({len(variance_imgs)})."
        )

    if isinstance(mask, (NiftiMasker, SurfaceMasker)):
        masker = mask.fit()
    elif mask is None:
        if isinstance(contrast_imgs[0], SurfaceImage):
            masker = SurfaceMasker().fit(contrast_imgs[0])
        else:
            masker = NiftiMasker().fit(contrast_imgs)
    elif isinstance(mask, SurfaceImage):
        masker = SurfaceMasker(mask_img=mask).fit(contrast_imgs[0])
    else:
        masker = NiftiMasker(mask_img=mask).fit()

    variances = np.array(
        [masker.transform(vi).squeeze() for vi in variance_imgs]
    )
    contrasts = np.array(
        [masker.transform(ci).squeeze() for ci in contrast_imgs]
    )

    if dofs is not None:
        if len(dofs) != n_runs:
            raise ValueError(
                f"The number of degrees of freedom ({len(dofs)}) "
                f"differs from the number of contrast images ({n_runs})."
            )
    else:
        dofs = [100] * n_runs

    (
        fixed_fx_contrast,
        fixed_fx_variance,
        fixed_fx_stat,
        fixed_fx_z_score,
    ) = _compute_fixed_effects_params(
        contrasts, variances, precision_weighted, dofs
    )

    fixed_fx_contrast_img = masker.inverse_transform(fixed_fx_contrast)
    fixed_fx_variance_img = masker.inverse_transform(fixed_fx_variance)
    fixed_fx_stat_img = masker.inverse_transform(fixed_fx_stat)
    fixed_fx_z_score_img = masker.inverse_transform(fixed_fx_z_score)
    warn(
        category=DeprecationWarning,
        message="The behavior of this function will be "
        "changed in release 0.13 to have an additional"
        "return value 'fixed_fx_z_score_img'  by default. "
        "Please set return_z_score to True.",
    )
    if return_z_score:
        return (
            fixed_fx_contrast_img,
            fixed_fx_variance_img,
            fixed_fx_stat_img,
            fixed_fx_z_score_img,
        )
    else:
        return fixed_fx_contrast_img, fixed_fx_variance_img, fixed_fx_stat_img


def _compute_fixed_effects_params(
    contrasts, variances, precision_weighted, dofs
):
    """Compute the fixed effects t/F-statistic, contrast, variance, \
    given arrays of effects and variance.
    """
    tiny = 1.0e-16
    contrasts, variances = np.asarray(contrasts), np.asarray(variances)
    variances = np.maximum(variances, tiny)

    if precision_weighted:
        weights = 1.0 / variances
        fixed_fx_variance = 1.0 / np.sum(weights, 0)
        fixed_fx_contrasts = np.sum(contrasts * weights, 0) * fixed_fx_variance
    else:
        fixed_fx_variance = np.mean(variances, 0) / len(variances)
        fixed_fx_contrasts = np.mean(contrasts, 0)
    dim = 1
    stat_type = "t"
    fixed_fx_contrasts_ = fixed_fx_contrasts
    if len(fixed_fx_contrasts.shape) == 2:
        dim = fixed_fx_contrasts.shape[0]
        if dim > 1:
            stat_type = "F"
    else:
        fixed_fx_contrasts_ = fixed_fx_contrasts

    con = Contrast(
        effect=fixed_fx_contrasts_,
        variance=fixed_fx_variance,
        dim=dim,
        dof=np.sum(dofs),
        stat_type=stat_type,
    )
    fixed_fx_z_score = con.z_score()
    fixed_fx_stat = con.stat_

    return (
        fixed_fx_contrasts,
        fixed_fx_variance,
        fixed_fx_stat,
        fixed_fx_z_score,
    )
