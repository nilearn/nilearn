""" Small utilities to inspect classes
"""

import inspect


def get_params(cls, instance, ignore=None):
    """ Retrieve the parameters corresponding to the class 'cls' for
    object 'instance'.

    Parameters
    ==========
    cls: class
        The class that gives us the list of parameters we are interested
        in

    instance: object, instance of BaseEstimator
        The object that gives us the values of the parameters

    ignore: None of list of strings
        The names of the parameters that are not returned.
    """

    _ignore = set(('memory', 'memory_level', 'verbose', 'copy'))
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


def retrieve_scope():
    try:
        caller_frame = inspect.currentframe().f_back.f_back
        if 'self' in caller_frame.f_locals:
            caller_name = caller_frame.f_locals['self'].__class__.__name__
            caller_name = '%s.%s' % (caller_name,
                                    caller_frame.f_code.co_name)
            #caller_name = caller_frame.f_code.co_name
            return caller_name
    except Exception:
        return 'Unknown'
