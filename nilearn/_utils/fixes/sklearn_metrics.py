from sklearn.metrics import get_scorer
from sklearn.externals import six
from warnings import warn


def _passthrough_scorer(estimator, *args, **kwargs):
    """Function that wraps estimator.score"""
    return estimator.score(*args, **kwargs)


def check_scoring(estimator, scoring=None, allow_none=False):
    """Determine scorer from user options.
    A TypeError will be thrown if the estimator cannot be scored.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    allow_none : boolean, optional, default: False
        If no scoring is specified and the estimator has no score function, we
        can either return None or raise an exception.
    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    """
    if not hasattr(estimator, 'fit'):
        raise TypeError("estimator should be an estimator implementing "
                        "'fit' method, %r was passed" % estimator)
    if isinstance(scoring, six.string_types):
        return get_scorer(scoring)
    elif callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, '__module__', None)
        if hasattr(module, 'startswith') and \
           module.startswith('sklearn.metrics.') and \
           not module.startswith('sklearn.metrics.scorer') and \
           not module.startswith('sklearn.metrics.tests.'):
            raise ValueError('scoring value %r looks like it is a metric '
                             'function rather than a scorer. A scorer should '
                             'require an estimator as its first parameter. '
                             'Please use `make_scorer` to convert a metric '
                             'to a scorer.' % scoring)
        return get_scorer(scoring)
    elif scoring is None:
        if hasattr(estimator, 'score'):
            return _passthrough_scorer
        elif allow_none:
            return None
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                "have a 'score' method. The estimator %r does not."
                % estimator)
    elif isinstance(scoring, Iterable):
        raise ValueError("For evaluating multiple scores, use "
                         "sklearn.model_selection.cross_validate instead. "
                         "{0} was passed.".format(scoring))
    else:
        raise ValueError("scoring value should either be a callable, string or"
                         " None. %r was passed" % scoring)
