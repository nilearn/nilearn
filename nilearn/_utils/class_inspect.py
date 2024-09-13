"""Small utilities to inspect classes."""

from sklearn.utils.estimator_checks import (
    check_classifiers_one_label_sample_weights,
    check_decision_proba_consistency,
    check_estimator as sklearn_check_estimator,
    check_estimator_get_tags_default_keys,
    check_estimators_partial_fit_n_features,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_non_transformer_estimators_n_iter,
    check_parameters_default_constructible,
    check_set_params,
)

VALID_CHECKS = [
    check_no_attributes_set_in_init,
    check_estimator_get_tags_default_keys,
    check_classifiers_one_label_sample_weights,
    check_estimators_partial_fit_n_features,
    check_non_transformer_estimators_n_iter,
    check_decision_proba_consistency,
    check_parameters_default_constructible,
    check_get_params_invariance,
    check_set_params,
]


def check_estimator(estimator=None):
    """Check compatibility with scikit-learn estimators.

    As some of nilearn estimators cannot fit numpy arrays,
    we cannot directly use
    sklearn.utils.estimator_checks.check_estimator.

    So this is a home made implementation that:
    - run the checks from sklearn we know are valid
    - make sure that those that fail keep on failing
    - implements some of our own checks
    """
    for check in sklearn_check_estimator(
        estimator=estimator, generate_only=True
    ):
        if check in VALID_CHECKS:
            check(estimator)


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
