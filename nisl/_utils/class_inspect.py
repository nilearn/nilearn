""" Small utilities to inspect classes
"""

from sklearn.base import BaseEstimator
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

    Returns
    =======
    params: dict
        The dict of parameters
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


def enclosing_scope_name(ensure_estimator=True, stack_level=2):
    """ Find the name of the enclosing scope

    Parameters
    ==========
    ensure_estimator: boolean, default: True
        If true, find the enclosing object deriving from 'BaseEstimator'
    stack_level: integer, default 2
        If ensure_estimator is not True, stack_level quantifies the
        number of frame we will go up.
    """
    try:
        frame = inspect.currentframe()
        if not ensure_estimator:
            for _ in range(stack_level):
                frame = frame.f_back
        else:
            while True:
                frame = frame.f_back
                if not 'self' in frame.f_locals:
                    continue
                if not isinstance(frame.f_locals['self'], BaseEstimator):
                    continue
                break
        if 'self' in frame.f_locals:
            caller_name = frame.f_locals['self'].__class__.__name__
            caller_name = '%s.%s' % (caller_name,
                                    frame.f_code.co_name)
        else:
            caller_name = frame.f_code.co_name

        return caller_name
    except Exception:
        return 'Unknown'
