"""Nilearn tags for estimators.

These tags extend the sklearn estimator tags
(https://scikit-learn.org/stable/developers/develop.html#estimator-tags)
to say which inputs an estimator accepts:

- ``niimg_like``: Niimg-like inputs
  (path to a ``.nii`` or ``.nii.gz`` file, or a Nifti image object)
- ``surf_img``: :class:`~nilearn.surface.SurfaceImage` inputs

Two ``estimator_type`` values are also used that sklearn does not define:
``"masker"`` and ``"glm"``.

Estimators declare their tags by overriding ``__sklearn_tags__``
and setting ``tags.input_tags = InputTags(...)``.
"""

from dataclasses import dataclass
from typing import Any

from sklearn.utils import InputTags as SkInputTags


@dataclass
class InputTags(SkInputTags):
    """Tags for the input data.

    Nilearn version of :class:`sklearn.utils.InputTags`
    with two extra tags for neuroimaging inputs.
    All other parameters are those of :class:`sklearn.utils.InputTags`.

    .. nilearn_versionadded:: 0.15.0

    Parameters
    ----------
    niimg_like : :obj:`bool`, default=True
        Whether the estimator accepts Niimg-like inputs.

    surf_img : :obj:`bool`, default=False
        Whether the estimator accepts
        :class:`~nilearn.surface.SurfaceImage` inputs.
    """

    # same as base input tags of
    # sklearn.utils.InputTags
    one_d_array: bool = False
    two_d_array: bool = True
    three_d_array: bool = False
    sparse: bool = False
    categorical: bool = False
    string: bool = False
    dict: bool = False
    positive_only: bool = False
    allow_nan: bool = False
    pairwise: bool = False

    # nilearn specific things

    # estimator accepts for str, Path to .nii[.gz] file
    # or NiftiImage object
    niimg_like: bool = True
    # estimator accepts SurfaceImage object
    surf_img: bool = False


def get_input_tag(estimator: Any, tag: str) -> bool:
    """Get the value of an input tag of an estimator.

    Parameters
    ----------
    estimator : estimator instance
        Estimator with a ``__sklearn_tags__`` method.

    tag : :obj:`str`
        Name of the input tag, for example ``"surf_img"``.

    Returns
    -------
    :obj:`bool`
        Value of the tag, or ``False`` if the estimator
        has no ``__sklearn_tags__`` method or no such tag.
    """
    if not hasattr(estimator, "__sklearn_tags__"):
        return False
    tags = estimator.__sklearn_tags__()
    return getattr(tags.input_tags, tag, False)


def is_masker(estimator: Any) -> bool:
    """Return True if the estimator is a masker."""
    if not hasattr(estimator, "__sklearn_tags__"):
        return False
    return estimator.__sklearn_tags__().estimator_type == "masker"


def is_glm(estimator: Any) -> bool:
    """Return True if the estimator is a GLM."""
    if not hasattr(estimator, "__sklearn_tags__"):
        return False
    return estimator.__sklearn_tags__().estimator_type == "glm"


def accepts_volume(estimator: Any) -> bool:
    """Return True if the estimator accepts Niimg-like inputs."""
    return get_input_tag(estimator, "niimg_like")


def accepts_surface(estimator: Any) -> bool:
    """Return True if the estimator accepts SurfaceImage inputs."""
    return get_input_tag(estimator, "surf_img")
