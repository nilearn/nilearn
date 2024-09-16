"""Small utilities to inspect classes."""

from sklearn.utils.estimator_checks import (
    check_decision_proba_consistency,
    check_estimator as sklearn_check_estimator,
    check_estimator_get_tags_default_keys,
    check_estimators_partial_fit_n_features,
    check_get_params_invariance,
    check_non_transformer_estimators_n_iter,
    check_set_params,
)

VALID_CHECKS = [
    x.__name__
    for x in [
        check_estimator_get_tags_default_keys,
        check_estimators_partial_fit_n_features,
        check_non_transformer_estimators_n_iter,
        check_decision_proba_consistency,
        check_get_params_invariance,
        check_set_params,
    ]
]

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
    """Check compatibility with scikit-learn estimators.

    As some of nilearn estimators cannot fit numpy arrays,
    we cannot directly use
    sklearn.utils.estimator_checks.check_estimator.

    So this is a home made implementation that yields an estimator instance
    along with a
    - valid check from sklearn: those should stay valid
    - or an invalid check that is known to fail.

    If new 'valid' checks are added to scikit-learn,
    then tests marked as xfail will start passing.

    Parameters
    ----------
    estimator : estimator object or list of estimator object
        Estimator instance to check.

    valid : bool, default=True
        Whether to return only the valid checks or not.

    extra_valid_checks : list of strings
        Names of checks to be tested as valid for this estimator.
    """
    if extra_valid_checks is None:
        valid_checks = VALID_CHECKS
    else:
        valid_checks = VALID_CHECKS + extra_valid_checks

    if not isinstance(estimator, list):
        estimator = [estimator]
    for est in estimator:
        for e, check in sklearn_check_estimator(
            estimator=est, generate_only=True
        ):
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

    params = dict()
    for param_name in param_names:
        if param_name in _ignore:
            continue
        if hasattr(instance, param_name):
            params[param_name] = getattr(instance, param_name)

    return params
