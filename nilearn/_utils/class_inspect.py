""" Small utilities to inspect classes
"""

from sklearn.base import BaseEstimator
import inspect

from .exceptions import AuthorizedException


def get_params(cls, instance, ignore=None):
    """ Retrieve the initialization parameters corresponding to a class

    This helper function retrieves the parameters of function __init__ for
    class 'cls' and returns the value for these parameters in object
    'instance'. When using a composition pattern (e.g. with a NiftiMasker
    class), it is useful to forward parameters from one instance to another.

    Parameters
    ==========
    cls: class
        The class that gives us the list of parameters we are interested
        in

    instance: object, instance of BaseEstimator
        The object that gives us the values of the parameters

    ignore: None or list of strings
        Names of the parameters that are not returned.

    Returns
    =======
    params: dict
        The dict of parameters
    """

    _ignore = set(('memory', 'memory_level', 'verbose', 'copy', 'n_jobs'))
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
    """ Find the name of the enclosing scope for debug output purpose

    Use inspection to climb up the stack until the calling object. This is
    typically used to get the estimator at the origin of a functional call
    for debug print purpose.

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
    except AuthorizedException:
        return 'Unknown'
