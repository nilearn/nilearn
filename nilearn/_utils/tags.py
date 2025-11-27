"""Nilearn tags for estimators.

These tags override or extends some of the sklearn tags.

With those tags we can specify if one of Nilearn's 'estimator'
(those include our maskers)
has certain characteristics or expected behavior.
For example if the estimator can accept nifti and / or surface images
during fitting.

This is mostly used internally to run some checks on our API
and its behavior.

See the sklearn documentation for more details on tags
https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
"""

from dataclasses import dataclass
from typing import Any

from packaging.version import parse
from sklearn import __version__ as sklearn_version

SKLEARN_LT_1_6 = parse(sklearn_version).release[1] < 6

if SKLEARN_LT_1_6:

    def tags(
        niimg_like=True,
        surf_img=False,
        **kwargs,
    ):
        """Add nilearn tags to estimator.

        See also: InputTags

        TODO (sklearn >= 1.6.0) remove
        """
        X_types = kwargs.get("X_types", [])
        X_types.append("2darray")
        if niimg_like:
            X_types.append("niimg_like")
        if surf_img:
            X_types.append("surf_img")
        X_types = list(set(X_types))

        return dict(X_types=X_types, **kwargs)

else:
    from sklearn.utils import InputTags as SkInputTags

    @dataclass
    class InputTags(SkInputTags):
        """Tags for the input data.

        Nilearn version of sklearn.utils.InputTags
        https://scikit-learn.org/1.6/modules/generated/sklearn.utils.InputTags.html#sklearn.utils.InputTags
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


def get_tag(estimator: Any, tag: str) -> bool:
    if not hasattr(estimator, "__sklearn_tags__"):
        return False
    tags = estimator.__sklearn_tags__()
    # TODO (sklearn >= 1.6.0) simplify
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        return tag in tags["X_types"]
    else:
        return getattr(tags.input_tags, tag, False)


def is_masker(estimator: Any) -> bool:
    if SKLEARN_LT_1_6:
        return getattr(estimator, "_estimator_type", "") == "masker"
    if not hasattr(estimator, "__sklearn_tags__"):
        return False
    return estimator.__sklearn_tags__().estimator_type == "masker"


def is_glm(estimator: Any) -> bool:
    if SKLEARN_LT_1_6:
        return getattr(estimator, "_estimator_type", "") == "glm"
    if not hasattr(estimator, "__sklearn_tags__"):
        return False
    return estimator.__sklearn_tags__().estimator_type == "glm"


def accept_niimg_input(estimator: Any) -> bool:
    return get_tag(estimator, "niimg_like")


def accept_surf_img_input(estimator: Any) -> bool:
    return get_tag(estimator, "surf_img")
