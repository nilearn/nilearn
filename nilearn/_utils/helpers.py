import functools
import operator
import os
import warnings


def rename_parameters(replacement_params,
                      end_version='future',
                      lib_name='Nilearn',
                      ):
    """Use this decorator to deprecate & replace specified parameters \
    in the decorated functions and methods without changing \
    function definition or signature.

    Parameters
    ----------
    replacement_params : Dict[string, string]
        Dict where the key-value pairs represent the old parameters
        and their corresponding new parameters.
        Example: {old_param1: new_param1, old_param2: new_param2,...}

    end_version : str {'future' | 'next' | <version>}, default='future'
        Version when using the deprecated parameters will raise an error.
        For informational purpose in the warning text.

    lib_name : str, default='Nilearn'
        Name of the library to which the decoratee belongs.
        For informational purpose in the warning text.

    """
    def _replace_params(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _warn_deprecated_params(replacement_params, end_version, lib_name,
                                    kwargs
                                    )
            kwargs = _transfer_deprecated_param_vals(replacement_params,
                                                     kwargs
                                                     )
            return func(*args, **kwargs)

        return wrapper
    return _replace_params


def _warn_deprecated_params(replacement_params, end_version, lib_name, kwargs):
    """Raise warnings about deprecated parameters, \
    for the decorator replace_parameters().

    Parameters
    ----------
    replacement_params : Dict[str, str]
        Dictionary of old_parameters as keys with replacement parameters
        as their corresponding values.

    end_version : str
        The version where use of the deprecated parameters will raise an error.
        For informational purpose in the warning text.

    lib_name : str
        Name of the library. For informational purpose in the warning text.

    kwargs : Dict[str, any]
        Dictionary of all the keyword args passed on the decorated function.

    """
    used_deprecated_params = set(kwargs).intersection(replacement_params)
    for deprecated_param_ in used_deprecated_params:
        replacement_param = replacement_params[deprecated_param_]
        param_deprecation_msg = (
            f'The parameter "{deprecated_param_}" '
            f'will be removed in {end_version} release of {lib_name}. '
            f'Please use the parameter "{replacement_param}" instead.')
        warnings.warn(category=DeprecationWarning,
                      message=param_deprecation_msg,
                      stacklevel=3)


def _transfer_deprecated_param_vals(replacement_params, kwargs):
    """Reassigns new parameters \
    the values passed to their corresponding deprecated parameters \
    for the decorator replace_parameters().

    Parameters
    ----------
    replacement_params : Dict[str, str]
        Dictionary of old_parameters as keys with replacement parameters
        as their corresponding values.

    kwargs : Dict[str, any]
        Dictionary of all the keyword args passed on the decorated function.

    Returns
    -------
    kwargs : Dict[str, any]
        Dictionary of all the keyword args to be passed on
        to the decorated function, with old parameter names
        replaced by new parameters, with their values intact.

    """
    for old_param, new_param in replacement_params.items():
        old_param_val = kwargs.setdefault(old_param, None)
        if old_param_val is not None:
            kwargs[new_param] = old_param_val
        kwargs.pop(old_param)
    return kwargs


def remove_parameters(removed_params,
                      reason,
                      end_version='future'):
    """Use this decorator to deprecate \
    but not renamed parameters in the decorated functions and methods.

    Parameters
    ----------
    removed_params : list[string]
        List of old parameters to be removed.
        Example: [old_param1, old_param2, ...]

    reason : str
        Detailed reason of deprecated parameter and alternative solutions.

    end_version : str {'future' | 'next' | <version>}, default='future'
        Version when using the deprecated parameters will raise an error.
        For informational purpose in the warning text.

    """
    def _remove_params(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            found = set(removed_params).intersection(kwargs)
            if found:
                message = (f'Parameter(s) {", ".join(found)} '
                           f'will be removed in version {end_version}; '
                           f'{reason}')
                warnings.warn(category=DeprecationWarning,
                              message=message,
                              stacklevel=3)
            return func(*args, **kwargs)
        return wrapper
    return _remove_params


def stringify_path(path):
    """Convert path-like objects to string.

    This is used to allow functions expecting string filesystem paths to accept
    objects using `__fspath__` protocol.

    Parameters
    ----------
    path : str or path-like object

    Returns
    -------
    str

    """
    return path.__fspath__() if isinstance(path, os.PathLike) else path


VERSION_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def compare_version(version_a, operator, version_b):
    """Compare two version strings via a user-specified operator.

    Note: This function is inspired from MNE-Python.
    See https://github.com/mne-tools/mne-python/blob/main/mne/fixes.py

    Parameters
    ----------
    version_a : :obj:`str`
        First version string.

    operator : {'==', '!=','>', '<', '>=', '<='}
        Operator to compare ``version_a`` and ``version_b`` in the form of
        ``version_a operator version_b``.

    version_b : :obj:`str`
        Second version string.

    Returns
    -------
    result : :obj:`bool`
        The result of the version comparison.

    """
    from packaging.version import parse

    if operator not in VERSION_OPERATORS:
        error_msg = "'compare_version' received an unexpected operator "
        raise ValueError(error_msg + operator + ".")
    return VERSION_OPERATORS[operator](parse(version_a), parse(version_b))


def is_plotly_installed():
    """Check if plotly is installed."""
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        return False
    return True


def is_kaleido_installed():
    """Check if kaleido is installed."""
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False
    return True
