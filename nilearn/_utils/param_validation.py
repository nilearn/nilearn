"""
Utilities to check for valid parameters
"""

import numbers
import warnings

from .compat import _basestring


def check_threshold(threshold, data, percentile_calculate, name='threshold'):
    """ Checks if the given threshold is in correct format and within the limit.

    If necessary, this function also returns score of the data calculated based
    upon the given specific percentile function.
    Note: This is only for threshold as string.

    Parameters
    ----------
    threshold: a float value or a real number or a percentage in string.
        If threshold is a float value, it should be within the range of the
        maximum intensity value of the data.
        If threshold is a percentage expressed in a string it must finish with a
        percent sign like "99.7%" or just a real number as 99.
    data: ndarray
        an array of the input masked data.
    percentile_calculate: a percentile function {scoreatpercentile, fastabspercentile}
        define the name of a specific percentile function to use it to
        calculate the score on the data.
    name: string, optional
        A string just used for representing the name of the threshold for a precise
        error message.

    Returns
    -------
    threshold: a number
        returns the score of the percentile on the data or
        returns threshold as it is if given threshold is not a string percentile.
    """
    if isinstance(threshold, _basestring):
        message = ('If "{0}" is given as string it '
                   'should be a number followed by the percent '
                   'sign, e.g. "25.3%"').format(name)
        if not threshold.endswith('%'):
            raise ValueError(message)

        try:
            percentile = float(threshold[:-1])
        except ValueError as exc:
            exc.args += (message, )
            raise

        threshold = percentile_calculate(data, percentile)
    elif isinstance(threshold, numbers.Real):
        # checks whether given float value exceeds the maximum
        # value of the image data
        value_check = abs(data).max()
        if abs(threshold) > value_check:
            warnings.warn("The given float value must not exceed %d. "
                          "But, you have given threshold=%s " % (value_check,
                                                                 threshold))
    else:
        raise TypeError('%s should be either a number '
                        'or a string finishing with a percent sign' % (name, ))
    return threshold
