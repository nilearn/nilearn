"""Cross-validation with image inputs."""

import numpy as np
from sklearn.model_selection import cross_val_score

from nilearn.image import iter_img
from nilearn.nilearn_typing import NiimgLike
from nilearn.surface import SurfaceImage


def cross_val_decoder_score(
    estimator,
    imgs,
    y,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
) -> np.ndarray:
    """Evaluate an image-based estimator with outer cross-validation.

    Adapt image inputs to :func:`sklearn.model_selection.cross_val_score`.
    Each fold fits a fresh clone of the estimator on training images and
    evaluates it on held-out images.

    .. nilearn_versionadded:: 0.15.0

    Parameters
    ----------
    estimator : estimator object
        An estimator accepting images in its ``fit`` and prediction methods,
        such as :class:`~nilearn.decoding.Decoder`,
        :class:`~nilearn.decoding.DecoderRegressor`,
        :class:`~nilearn.decoding.FREMClassifier`, or
        :class:`~nilearn.decoding.FREMRegressor`.

    imgs : Niimg-like object, :obj:`~nilearn.surface.SurfaceImage`, or iterable
        A 4D image or its path, an iterable of 3D images or their paths,
        a surface image, or an iterable of surface images.
        Each volume or surface sample corresponds to one element of ``y``.

    y : array-like of shape (n_samples,)
        Target values, in the same order as the image samples.

    groups : array-like of shape (n_samples,) or None, default=None
        Group labels for the outer cross-validation splitter. To also pass
        groups to the estimator's ``fit``, use ``params={"groups": groups}``.
        For grouped inner cross-validation, set the estimator's ``cv`` to a
        group-aware splitter.
        When scikit-learn metadata routing is enabled, pass groups only in
        ``params`` and configure the estimator's metadata requests.

    scoring : :obj:`str`, callable, or None, default=None
        A scikit-learn scorer name or a callable with signature
        ``scorer(estimator, imgs, y)`` returning a single score.
        If None, use the estimator's ``score`` method. For decoders this uses
        their own ``scoring`` parameter.

    cv : :obj:`int`, cross-validation splitter, iterable, or None, default=None
        The outer cross-validation scheme. An integer specifies the number
        of folds; None uses five folds. Classification with binary or
        multiclass targets uses stratified folds, otherwise K-fold is used.
        A splitter or an iterable of (train, test) index pairs is also
        accepted. Splitters receive a list of image samples as ``X``.

    n_jobs : :obj:`int` or None, default=None
        Number of outer folds to evaluate in parallel. None uses one job
        unless a joblib parallel configuration specifies otherwise.
        -1 uses all processors. Set the decoder's own ``n_jobs=1`` when
        parallelizing outer folds to avoid nested parallelism.

    verbose : :obj:`int`, default=0
        Verbosity of scikit-learn's cross-validation computation.

    params : :obj:`dict` or None, default=None
        Metadata passed to scikit-learn's cross-validation machinery.
        With metadata routing disabled, parameters are passed to ``fit``.
        With routing enabled, they may also be routed to the splitter and
        scorer. Sample-aligned parameters are subset for each fold.

    pre_dispatch : :obj:`int` or :obj:`str`, default="2*n_jobs"
        Number of jobs dispatched ahead of execution. Reducing this value
        can limit memory use when evaluating large images in parallel.

    error_score : :obj:`float` or "raise", default=np.nan
        Score assigned when fitting or scoring fails. If "raise", propagate
        the exception. Numerical values produce a warning; if all fits
        fail, scikit-learn raises an error.

    Returns
    -------
    scores : :obj:`numpy.ndarray` of shape (n_splits,)
        One held-out score per outer split, in splitter order.

    See Also
    --------
    sklearn.model_selection.cross_val_score : Cross-validation on sample lists
        or arrays.
    sklearn.model_selection.cross_validate : Evaluate multiple metrics and
        optionally return fitted estimators.
    nilearn.image.iter_img : Iterate over individual image samples.

    Notes
    -----
    Decoders already perform internal cross-validation for model selection
    and aggregation. This function adds an outer loop: the decoder's
    internal cross-validation runs separately within each outer training
    fold. These held-out scores differ from ``decoder.cv_scores_``, which
    records internal model-selection scores. Nested cross-validation can
    therefore be substantially more expensive than a single decoder fit.

    Image conversion does not fit a masker or any other preprocessing.
    The cloned estimator fits its masker and model on each training fold.
    Preprocessing follows the estimator's existing transform behavior.
    Any masks or preprocessed images supplied by the caller should be
    defined independently of the held-out data.

    Scoring, validation, metadata routing, and parallel execution follow
    scikit-learn's ``cross_val_score`` conventions. Existing sample lists
    can also be passed directly to that function.

    Examples
    --------
    >>> import numpy as np
    >>> from nibabel import Nifti1Image
    >>> from nilearn.decoding import Decoder, cross_val_decoder_score
    >>> rng = np.random.default_rng(0)
    >>> img = Nifti1Image(rng.normal(size=(5, 5, 5, 30)), np.eye(4))
    >>> mask = Nifti1Image(np.ones((5, 5, 5)), np.eye(4))
    >>> decoder = Decoder(mask=mask, cv=2, screening_percentile=None)
    >>> scores = cross_val_decoder_score(
    ...     decoder, img, np.tile([0, 1], 15), cv=3, scoring="accuracy"
    ... )
    >>> scores.shape
    (3,)
    """
    if isinstance(imgs, SurfaceImage):
        imgs = [imgs]
    elif isinstance(imgs, NiimgLike):
        imgs = iter_img(imgs)

    imgs = list(imgs)
    if imgs and isinstance(imgs[0], SurfaceImage):
        # iter_img reshapes 1D surface data in place. Keep its container
        # separate from the caller's, without copying the data arrays.
        imgs = [
            sample
            for img in imgs
            for sample in iter_img(
                SurfaceImage(img.mesh, dict(img.data.parts))
            )
        ]

    return cross_val_score(
        estimator,
        imgs,
        y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
    )
