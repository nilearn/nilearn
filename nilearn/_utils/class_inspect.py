"""Small utilities to inspect classes."""

import numpy as np
from nibabel import Nifti1Image
from sklearn import __version__ as sklearn_version
from sklearn import clone
from sklearn.utils.estimator_checks import (
    check_estimator as sklearn_check_estimator,
)

from nilearn._utils import compare_version
from nilearn._utils.exceptions import DimensionError

# List of sklearn estimators checks that are valid
# for all nilearn estimators.
VALID_CHECKS = [
    "check_decision_proba_consistency",
    "check_estimator_cloneable",
    "check_estimators_partial_fit_n_features",
    "check_estimator_repr",
    "check_estimator_tags_renamed",
    "check_get_params_invariance",
    "check_mixin_order",
    "check_non_transformer_estimators_n_iter",
    "check_parameters_default_constructible",
    "check_set_params",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
    # Nilearn checks
    "check_masker_fitted",
    "check_nifti_masker_fit_list_3d",
    "check_nifti_masker_fit_with_3d_mask",
    "check_nifti_masker_fit_with_4d_mask",
    "check_nifti_masker_fit_with_empty_mask",
]

if compare_version(sklearn_version, ">", "1.5.2"):
    VALID_CHECKS.append("check_valid_tag_types")
else:
    VALID_CHECKS.append("check_estimator_get_tags_default_keys")


# TODO / TOCHECK: with sklearn >= 1.6
# some of those checks should be skipped 'automatically'
# by sklearn
# and could be removed from this list.
CHECKS_TO_SKIP_IF_IMG_INPUT = {
    "check_estimator_sparse_array",
    "check_estimator_sparse_data",
    "check_estimator_sparse_matrix",
    "check_estimator_sparse_tag",
    "check_f_contiguous_array_estimator",
    "check_fit1d",
    "check_fit2d_1feature",
    "check_fit2d_1sample",
    "check_fit2d_predict1d",
}

# TODO
# remove when bumping to sklearn >= 1.3
try:
    from sklearn.utils.estimator_checks import (
        check_classifiers_one_label_sample_weights,
    )

    VALID_CHECKS.append(check_classifiers_one_label_sample_weights.__name__)
except ImportError:
    ...


def check_estimator(estimator=None, valid=True, extra_valid_checks=None):
    """Yield a valid or invalid scikit-learn estimators check.

    As some of Nilearn estimators do not comply
    with sklearn recommendations
    (cannot fit Numpy arrays, do input validation in the constructor...)
    we cannot directly use
    sklearn.utils.estimator_checks.check_estimator.

    So this is a home made generator that yields an estimator instance
    along with a
    - valid check from sklearn: those should stay valid
    - or an invalid check that is known to fail.

    If new 'valid' checks are added to scikit-learn,
    then tests marked as xfail will start passing.

    If estimator have some nilearn specific tags
    then some checks will skip rather than yield.

    See this section rolling-your-own-estimator in
    the scikit-learn doc for more info:
    https://scikit-learn.org/stable/developers/develop.html

    Parameters
    ----------
    estimator : estimator object or list of estimator object
        Estimator instance to check.

    valid : bool, default=True
        Whether to return only the valid checks or not.

    extra_valid_checks : list of strings
        Names of checks to be tested as valid for this estimator.
    """
    valid_checks = VALID_CHECKS
    if extra_valid_checks is not None:
        valid_checks = VALID_CHECKS + extra_valid_checks

    if not isinstance(estimator, list):
        estimator = [estimator]
    for est in estimator:
        for e, check in sklearn_check_estimator(
            estimator=est, generate_only=True
        ):
            tags = est._more_tags()

            niimg_input = False
            surf_img = False
            # TODO remove first if when dropping sklearn 1.5
            #  for sklearn >= 1.6 tags are always a dataclass
            if isinstance(tags, dict) and "X_types" in tags:
                niimg_input = "niimg_like" in tags["X_types"]
                surf_img = "surf_img" in tags["X_types"]
            else:
                niimg_input = getattr(tags.input_tags, "niimg_like", False)
                surf_img = getattr(tags.input_tags, "surf_img", False)

            if (
                niimg_input or surf_img
            ) and check.func.__name__ in CHECKS_TO_SKIP_IF_IMG_INPUT:
                continue

            if valid and check.func.__name__ in valid_checks:
                yield e, check, check.func.__name__
            if not valid and check.func.__name__ not in valid_checks:
                yield e, check, check.func.__name__

    if valid:
        for est in estimator:
            for e, check in nilearn_check_estimator(estimator=est):
                yield e, check, check.__name__


def nilearn_check_estimator(estimator):
    tags = estimator._more_tags()

    niimg_input = False
    is_masker = False
    # TODO remove first if when dropping sklearn 1.5
    #  for sklearn >= 1.6 tags are always a dataclass
    if isinstance(tags, dict) and "X_types" in tags:
        niimg_input = "niimg_like" in tags["X_types"]
        is_masker = "masker" in tags["X_types"]
    else:
        niimg_input = getattr(tags.input_tags, "niimg_like", False)
        is_masker = getattr(tags.input_tags, "masker", False)

    if is_masker:
        yield (clone(estimator), check_masker_fitted)

    if is_masker and niimg_input:
        yield (clone(estimator), check_nifti_masker_fit_list_3d)
        yield (clone(estimator), check_nifti_masker_fit_with_3d_mask)
        yield (clone(estimator), check_nifti_masker_fit_with_4d_mask)
        yield (clone(estimator), check_nifti_masker_fit_with_empty_mask)


def check_masker_fitted(estimator):
    """Check that transform() and inverse_transform() \
       fail for maskers if they have not been fitted.
    """
    import pytest

    from nilearn._utils.data_gen import generate_random_img

    # Failure should happen before the input type is determined
    # so we can pass nifti image to surface maskers.
    img_3d_rand_eye = generate_random_img(shape=(7, 8, 9), affine=np.eye(4))
    with pytest.raises(ValueError, match="has not been fitted."):
        estimator.transform(img_3d_rand_eye)

    # Failure should happen before the size of the input type is determined
    # so we can pass any array here.
    signals = np.ones((10, 11))
    with pytest.raises(ValueError, match="has not been fitted."):
        estimator.inverse_transform(signals)


def check_nifti_masker_fit_list_3d(estimator):
    """Check that list of 3D image can be fitted."""
    from nilearn.conftest import _img_3d_rand

    estimator.fit([_img_3d_rand(), _img_3d_rand()])


def check_nifti_masker_fit_with_3d_mask(estimator):
    """Check 3D mask can be used with nifti maskers.

    Mask can have different shape than fitted image.
    """
    from nilearn.conftest import _affine_eye, _img_3d_rand

    # TODO
    # (29, 30, 31) is used to match MAP_SHAPE in
    # nilearn/regions/tests/test_region_extractor.py
    # this test would fail for RegionExtractor otherwise
    mask = np.ones((29, 30, 31))
    mask_img = Nifti1Image(mask, affine=_affine_eye())

    estimator.mask_img = mask_img

    assert not hasattr(estimator, "mask_img_")

    estimator.fit([_img_3d_rand()])

    assert hasattr(estimator, "mask_img_")


def check_nifti_masker_fit_with_empty_mask(estimator):
    """Check mask that excludes all voxels raise an error."""
    import pytest

    from nilearn.conftest import _img_3d_rand, _img_3d_zeros

    estimator.mask_img = _img_3d_zeros()
    with pytest.raises(
        ValueError,
        match="The mask is invalid as it is empty: it masks all data",
    ):
        estimator.fit([_img_3d_rand()])


def check_nifti_masker_fit_with_4d_mask(estimator):
    """Check 4D mask cannot be used with nifti maskers."""
    import pytest

    from nilearn.conftest import _img_3d_rand, _img_4d_zeros

    with pytest.raises(DimensionError, match="Expected dimension is 3D"):
        estimator.mask_img = _img_4d_zeros()
        estimator.fit([_img_3d_rand()])


def get_params(cls, instance, ignore=None):
    """Retrieve the initialization parameters corresponding to a class.

    This helper function retrieves the parameters of function __init__ for
    class 'cls' and returns the value for these parameters in object
    'instance'. When using a composition pattern (e.g. with a NiftiMasker
    class), it is useful to forward parameters from one instance to another.

    Parameters
    ----------
    cls : class
        The class that gives us the list of parameters we are interested in.

    instance : object, instance of BaseEstimator
        The object that gives us the values of the parameters.

    ignore : None or list of strings
        Names of the parameters that are not returned.

    Returns
    -------
    params : dict
        The dict of parameters.

    """
    _ignore = {"memory", "memory_level", "verbose", "copy", "n_jobs"}
    if ignore is not None:
        _ignore.update(ignore)

    param_names = cls._get_param_names()

    params = {}
    for param_name in param_names:
        if param_name in _ignore:
            continue
        if hasattr(instance, param_name):
            params[param_name] = getattr(instance, param_name)

    return params
