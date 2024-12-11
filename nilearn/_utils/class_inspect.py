"""Small utilities to inspect classes."""

from dataclasses import dataclass

from packaging.version import parse
from sklearn import __version__ as sklearn_version
from sklearn.utils.estimator_checks import (
    check_estimator as sklearn_check_estimator,
)

from nilearn._utils import compare_version

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
    "check_set_params",
]

if compare_version(sklearn_version, ">", "1.5.2"):
    VALID_CHECKS.append("check_valid_tag_types")
else:
    VALID_CHECKS.append("check_estimator_get_tags_default_keys")


CHECKS_TO_SKIP_IF_NIIMG_INPUT = {
    "check_estimator_sparse_matrix",
    "check_estimator_sparse_array",
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
    then some checks will skipped rather than yielded.

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
            ) and check.func.__name__ in CHECKS_TO_SKIP_IF_NIIMG_INPUT:
                continue

            if valid and check.func.__name__ in valid_checks:
                yield e, check, check.func.__name__
            if not valid and check.func.__name__ not in valid_checks:
                yield e, check, check.func.__name__


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


ver = parse(sklearn_version)
if ver.release[1] >= 6:
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
else:

    def tags(niimg_like=True, surf_img=False, **kwargs):
        """Add nilearn tags to estimator.

        See also: InputTags

        TODO remove when dropping sklearn 1.5
        """
        X_types = []
        if "X_types" in kwargs:
            X_types = kwargs["X_types"]
        X_types.append("2darray")
        if niimg_like:
            X_types.append("niimg_like")
        if surf_img:
            X_types.append("surf_img")
        X_types = list(set(X_types))
        return dict(X_types=X_types, **kwargs)
