from sklearn.metrics import make_scorer, SCORERS
from sklearn.externals import six
from warnings import warn


def get_scorer(scoring):
    if isinstance(scoring, six.string_types):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            raise ValueError('%r is not a valid scoring value. '
                             'Valid options are %s'
                             % (scoring, sorted(SCORERS.keys())))
    else:
        scorer = scoring
    return scorer


def _passthrough_scorer(estimator, *args, **kwargs):
    """Function that wraps estimator.score"""
    return estimator.score(*args, **kwargs)


def check_scoring(estimator, scoring=None, allow_none=False, loss_func=None,
                  score_func=None, score_overrides_loss=False):
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
    has_scoring = not (scoring is None and loss_func is None and
                       score_func is None)
    if not hasattr(estimator, 'fit'):
        raise TypeError("estimator should a be an estimator implementing "
                        "'fit' method, %r was passed" % estimator)
    elif hasattr(estimator, 'predict') and has_scoring:
        scorer = None
        if loss_func is not None or score_func is not None:
            if loss_func is not None:
                warn("Passing a loss function is "
                     "deprecated and will be removed in 0.15. "
                     "Either use strings or score objects. "
                     "The relevant new parameter is called ''scoring''. ",
                     category=DeprecationWarning, stacklevel=2)
                scorer = make_scorer(loss_func, greater_is_better=False)
            if score_func is not None:
                warn("Passing function as ``score_func`` is "
                     "deprecated and will be removed in 0.15. "
                     "Either use strings or score objects. "
                     "The relevant new parameter is called ''scoring''.",
                     category=DeprecationWarning, stacklevel=2)
                if loss_func is None or score_overrides_loss:
                    scorer = make_scorer(score_func)
        else:
            scorer = get_scorer(scoring)
        return scorer
    elif hasattr(estimator, 'score'):
        return _passthrough_scorer
    elif not has_scoring:
        if allow_none:
            return None
        raise TypeError(
            "If no scoring is specified, the estimator passed should "
            "have a 'score' method. The estimator %r does not." % estimator)
    else:
        raise TypeError(
            "The estimator passed should have a 'score' or a 'predict' "
            "method. The estimator %r does not." % estimator)
