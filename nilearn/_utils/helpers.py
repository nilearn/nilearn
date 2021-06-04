import functools
import warnings


def rename_parameters(replacement_params,
                      end_version='future',
                      lib_name='Nilearn',
                      ):
    """Decorator to deprecate & replace specificied parameters
    in the decorated functions and methods without changing
    function definition or signature.

    Parameters
    ----------
    replacement_params : Dict[string, string]
        Dict where the key-value pairs represent the old parameters
        and their corresponding new parameters.
        Example: {old_param1: new_param1, old_param2: new_param2,...}

    end_version : str {'future' | 'next' | <version>}, optional
        Version when using the deprecated parameters will raise an error.
        For informational purpose in the warning text.
        Default='future'.

    lib_name : str, optional
        Name of the library to which the decoratee belongs.
        For informational purpose in the warning text.
        Default='Nilearn'.

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
    """For the decorator replace_parameters(), raises warnings about
    deprecated parameters.

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
            'The parameter "{}" will be removed in {} release of {}. '
            'Please use the parameter "{}" instead.'.format(deprecated_param_,
                                                            end_version,
                                                            lib_name,
                                                            replacement_param,
                                                            )
        )
        warnings.warn(category=FutureWarning,
                      message=param_deprecation_msg,
                      stacklevel=3)


def _transfer_deprecated_param_vals(replacement_params, kwargs):
    """For the decorator replace_parameters(), reassigns new parameters
    the values passed to their corresponding deprecated parameters.

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
    """Decorator to deprecate but not renamed parameters in the decorated
    functions and methods.

    Parameters
    ----------
    removed_params : list[string]
        List of old parameters to be removed.
        Example: [old_param1, old_param2, ...]

    reason : str
        Detailed reason of deprecated parameter and alternative solutions.

    end_version : str {'future' | 'next' | <version>}, optional
        Version when using the deprecated parameters will raise an error.
        For informational purpose in the warning text.
        Default='future'.

    """
    def _remove_params(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            found = set(removed_params).intersection(kwargs)
            if found:
                message = ('Parameter(s) {} will be removed in version {}; '
                           '{}'.format(', '.join(found),
                                       end_version, reason)
                           )
                warnings.warn(category=DeprecationWarning,
                              message=message,
                              stacklevel=3)
            return func(*args, **kwargs)
        return wrapper
    return _remove_params
