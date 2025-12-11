"""Utilities to discover nilearn objects."""

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils._testing import ignore_warnings

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.param_validation import check_parameter_in_allowed
from nilearn.decoding._mixin import _ClassifierMixin, _RegressorMixin
from nilearn.maskers._mixin import _MultiMixin
from nilearn.maskers.base_masker import BaseMasker, _BaseSurfaceMasker

ROOT = str(Path(__file__).parent.parent)  # nilearn package


_MODULE_TO_IGNORE = {
    "_utils",
    "conftest",
    "input_data",
    "tests",
}


def _skip_module(module_name: str):
    module_parts = module_name.split(".")
    return bool(
        any(part in _MODULE_TO_IGNORE for part in module_parts)
        or "._" in module_name
    )


def _is_abstract(c):
    """Determine if an object has abstract methods."""
    if not (hasattr(c, "__abstractmethods__")):
        return False
    return len(c.__abstractmethods__)


def _get_all_classes():
    """List all classes in the public API of Nilearn."""
    all_classes = []
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[ROOT], prefix="nilearn."
        ):
            if _skip_module(module_name):
                continue

            try:
                module = import_module(module_name)
            except ModuleNotFoundError:
                continue

            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls)
                for name, est_cls in classes
                if not name.startswith("_") and "sklearn" not in str(est_cls)
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    # get rid of abstract base classes
    all_classes = [c for c in all_classes if not _is_abstract(c[1])]

    return all_classes


def all_estimators(type_filter=None):
    """Get a list of all estimators from `nilearn`.

    This function crawls the module and gets all classes that inherit
    from sklearn BaseEstimator.
    Classes that are defined in test-modules are not included.

    Parameters
    ----------
    type_filter : {"classifier",  "cluster", "regressor", "masker", \
                  "multi_masker", "transformer"} \
                  or list of such strings, default=None
        Which kind of estimators should be returned.
        If ``None``, no filter is applied and all estimators are returned.
        Possible values are
        "classifier", "cluster", "regressor", "masker",
        "multi_masker", "transformer"
        to get estimators only of these specific types,
        or a list of these to get the estimators
        that fit at least one of the types.

    Returns
    -------
    estimators : :obj:`list` of :obj:`tuple`
        List of (name, class),
        where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    """
    # TODO: add GLM?
    allowed_filters = {
        "classifier": _ClassifierMixin,
        "cluster": ClusterMixin,
        "masker": (BaseMasker, _BaseSurfaceMasker),
        "multi_masker": _MultiMixin,
        "regressor": _RegressorMixin,
        "transformer": TransformerMixin,
    }

    all_classes = _get_all_classes()

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]

    if type_filter is not None:
        check_parameter_in_allowed(
            type_filter, allowed_filters.keys(), "type_filter"
        )

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
    functions : :obj:`list` of :obj:`tuple`
        List of ``(name, function)``,
        where ``name`` is the function name as string
        and ``function`` is the actual function.

    """
    all_functions = []
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[ROOT], prefix="nilearn."
        ):
            if _skip_module(module_name):
                continue

            try:
                module = import_module(module_name)
            except ModuleNotFoundError:
                continue

            if not hasattr(module, "__all__"):
                continue

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


def all_displays(type_filter=None):
    """Get a list of all 'displays' objects from `nilearn`.

    Parameters
    ----------
    type_filter : {"slicer",  "axe"} \
                  or list of such strings, default=None
        Which kind of display object should be returned.
        If ``None``, no filter is applied and all objects are returned.
        Possible values are
        "slicer",  "axe"
        to get only these specific types,
        or a list of display objects
        that fit at least one of the types.

    Returns
    -------
    displays : list of tuples
        List of (name, class), where ``name`` is the display class name as
        string and ``class`` is the actual type of the class.
    """
    if not is_matplotlib_installed():
        return []
    from nilearn.plotting.displays import BaseAxes, BaseSlicer

    allowed_filters = {
        "slicer": BaseSlicer,
        "axe": BaseAxes,
    }

    all_classes = _get_all_classes()

    displays = [
        c
        for c in all_classes
        if (issubclass(c[1], tuple(allowed_filters.values())))
    ]

    if type_filter is not None:
        check_parameter_in_allowed(
            type_filter, allowed_filters.keys(), "type_filter"
        )

        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_displays = []

        for name, this_class in allowed_filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_displays.extend(
                    [est for est in displays if issubclass(est[1], this_class)]
                )
        displays = filtered_displays

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend
    # to the 2nd item of the tuple
    return sorted(set(displays), key=itemgetter(0))
