"""Utilities to discover nilearn objects."""

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils._testing import ignore_warnings

from nilearn._utils.param_validation import check_parameter_in_allowed

_MODULE_TO_IGNORE = {
    "_utils",
    "conftest",
    "externals",
    "input_data",
    "tests",
}


def _skip_module(module_name: str):
    module_parts = module_name.split(".")
    return bool(
        any(part in _MODULE_TO_IGNORE for part in module_parts)
        or "._" in module_name
    )


def all_estimators(type_filter=None):
    """Get a list of all estimators from `nilearn`.

    This function crawls the module and gets all classes that inherit
    from sklearn BaseEstimator.
    Classes that are defined in test-modules are not included.

    Parameters
    ----------
    type_filter : {"classifier",  "cluster", "regressor", "masker",
                  "multi_masker", "transformer"} \
                  or list of such str, default=None
        Which kind of estimators should be returned.
        If None, no filter is applied and all estimators are returned.
        Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer'
        to get estimators only of these specific types,
        or a list of these to get the estimators
        that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    """
    # lazy import to avoid circular imports
    from nilearn.decoding._mixin import _ClassifierMixin, _RegressorMixin
    from nilearn.maskers._mixin import _MultiMixin
    from nilearn.maskers.base_masker import BaseMasker, _BaseSurfaceMasker

    # TODO: add GLM?
    allowed_filters = {
        "classifier": _ClassifierMixin,
        "cluster": ClusterMixin,
        "masker": (BaseMasker, _BaseSurfaceMasker),
        "multi_masker": _MultiMixin,
        "regressor": _RegressorMixin,
        "transformer": TransformerMixin,
    }

    if type_filter is not None and type_filter not in allowed_filters:
        check_parameter_in_allowed(
            type_filter, allowed_filters.keys(), "type_filter"
        )

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        return len(c.__abstractmethods__)

    all_classes = []
    root = str(Path(__file__).parent.parent)  # nilearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[root], prefix="nilearn."
        ):
            if _skip_module(module_name):
                continue

            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls)
                for name, est_cls in classes
                if not name.startswith("_") and "sklearn" not in str(est_cls)
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []

        for name, mixin in allowed_filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend
    # to the 2nd item of the tuple
    return sorted(set(estimators), key=itemgetter(0))


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    return not (
        not mod.startswith("nilearn.") or mod.endswith("estimator_checks")
    )


def all_functions():
    """Get a list of all functions from `nilearn`.

    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.

    """
    all_functions = []
    root = str(Path(__file__).parent.parent)  # nilearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[root], prefix="nilearn."
        ):
            if _skip_module(module_name):
                continue

            module = import_module(module_name)

            if not hasattr(module, "__all__"):
                continue

            print(module.__all__)

            functions = inspect.getmembers(module, _is_checked_function)
            functions = [
                (func.__name__, func)
                for name, func in functions
                if not name.startswith("_") and func.__name__ in module.__all__
            ]
            all_functions.extend(functions)

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_functions), key=itemgetter(0))
