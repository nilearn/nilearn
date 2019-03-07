import functools
import warnings


def replace_parameters(replacement_params, end_version, lib_name='Nilearn'):
    """
    Decorator to deprecate & replace specificied parameters
    in the decorated functions and methods.
    
    Add **kwargs as the last parameter in the decorated method/function.
    
    Parameters
    ----------
    replacement_params : Dict[string, string]
        Dict where the key-value pairs represent the old parameters
        and their corresponding new parameters.
        Example: {old_param1: new_param1, old_param2: new_param2,...}
        
    end_version : str
        Version when the deprecated parameters will cease functioning
        and no more warnings will be displayed.
        Default: None / 'future'
        Example: '0.6.0b', 'next'
        
    lib_name: str
        Name of the library to which the decoratee belongs.
        Default: 'Nilearn'
    """
    
    def _replace_params(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _warn_deprecated_params(replacement_params, end_version, lib_name, kwargs)
            kwargs = _transfer_deprecated_param_vals(replacement_params, kwargs)
            return func(*args, **kwargs)
        
        return wrapper
    return _replace_params


def _warn_deprecated_params(replacement_params, end_version, lib_name, kwargs):
    """ For the decorator replace_parameters(),
        raises warnings about deprecated parameters.
    """
    if end_version is None or end_version == 'future':
        lib_end_ver = 'a future {} version'.format(lib_name)
    elif end_version == 'next':
        lib_end_ver = 'the next {} version'.format(lib_name)
    else:
        lib_end_ver = '{} version {}'.format(lib_name, end_version)
    used_deprecated_params = set(kwargs).intersection(replacement_params)
    for deprecated_param_ in used_deprecated_params:
        replacement_param = replacement_params[deprecated_param_]
        param_deprecation_msg = (
            'The parameter "{}" will be removed in {}. '
            'Please use the parameter "{}" instead.'.format(deprecated_param_,
                                                            lib_end_ver,
                                                            replacement_param,
                                                            )
        )
        warnings.filterwarnings('always', message=param_deprecation_msg)
        warnings.warn(category=DeprecationWarning,
                      message=param_deprecation_msg,
                      stacklevel=3)


def _transfer_deprecated_param_vals(replacement_params, kwargs):
    """ For the decorator replace_parameters(), reassigns new parameters
    the values passed to their corresponding deprecated parameters.
    """
    for old_param, new_param in replacement_params.items():
        old_param_val = kwargs.setdefault(old_param, None)
        if old_param_val is not None:
            kwargs[new_param] = old_param_val
        kwargs.pop(old_param)
    return kwargs
