"""The :mod:`nilearn._utils` module provides utilities for developers."""

import pkgutil
import inspect
import warnings
from pathlib import Path
from importlib import import_module

from .niimg_conversions import (check_niimg,
                                check_niimg_3d,
                                concat_niimgs,
                                check_niimg_4d)

from .niimg import _repr_niimgs, copy_img, load_niimg

from .docs import fill_doc

from .numpy_conversions import as_ndarray

from .cache_mixin import CacheMixin

from .logger import _compose_err_msg
from nilearn._utils.helpers import rename_parameters, remove_parameters

__all__ = ['check_niimg', 'check_niimg_3d', 'concat_niimgs', 'check_niimg_4d',
           '_repr_niimgs', 'copy_img', 'load_niimg',
           'as_ndarray', 'CacheMixin', '_compose_err_msg', 'rename_parameters',
           'remove_parameters', 'fill_doc',
           ]


def all_modules(modules_to_ignore=None):
    """Get a list of all modules from nilearn.

    This function returns a list of all modules from Nilearn.

    Parameters
    ----------
    modules_to_ignore : List/Set of :obj:`str` or None
        List of modules to exclude from the listing.
        Default=None.

    Returns
    -------
    all_modules : List of :obj:`str`
        List of modules.
    """
    if modules_to_ignore is None:
        modules_to_ignore = {
            "data",
            "tests",
            "externals",
        }
    all_modules = []
    root = str(Path(__file__).parent.parent)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[root], prefix="nilearn."
        ):
            mod_parts = modname.split(".")
            if(
                any(part in modules_to_ignore for part in mod_parts)
                or "._" in modname
            ):
                continue
            all_modules.append(modname)
    return all_modules


def all_functions(return_private=False, modules_to_ignore=None):
    """Get a list of all functions from nilearn.

    This function returns a list of all functions defined in Nilearn.

    Parameters
    ----------
    return_private : :obj:`bool`
        Whether to return also private functions or not.
        Default=False.

    modules_to_ignore : List/Set of :obj:`str` or None
        List of modules to exclude from the listing.
        Default=None.

    Returns
    -------
    all_functions : List of Tuples (:obj:`str`, callable)
        List of functions. Each element is a length 2 tuple
        where the first element is the function name as a string,
        and the second element is the function itself.
    """
    all_functions = []
    for modname in all_modules(modules_to_ignore=modules_to_ignore):
        module = import_module(modname)
        functions = [
            (name, func) for name, func in inspect.getmembers(
                module, inspect.isfunction
            )
            if func.__module__ == module.__name__
        ]
        if not return_private:
            functions = [
                (name, func) for name, func in functions
                if not name.startswith("_")
            ]
        all_functions.extend(functions)
    return all_functions


def all_classes(return_private=False, modules_to_ignore=None):
    """Get a list of all classes from nilearn.

    This function returns a list of all classes defined in Nilearn.

    Parameters
    ----------
    return_private : :obj:`bool`
        Whether to return also private classes or not.
        Default=False.

    modules_to_ignore : List/Set of :obj:`str` or None
        List of modules to exclude from the listing.
        Default=None.

    Returns
    -------
    all_classes : List of Tuples (:obj:`str`, callable)
        List of classes. Each element is a length 2 tuple
        where the first element is the class name as a string,
        and the second element is the class itself.
    """
    all_classes = []
    for modname in all_modules(modules_to_ignore=modules_to_ignore):
        module = import_module(modname)
        classes = [
            (name, cls) for name, cls in inspect.getmembers(
                module, inspect.isclass
            )
            if cls.__module__ == module.__name__
        ]
        if not return_private:
            classes = [
                (name, cls) for name, cls in classes
                if not name.startswith("_")
            ]
        all_classes.extend(classes)
    return all_classes
