"""
Utilities to check for valid parameters
"""

import numbers
import warnings

from .compat import _basestring, izip


def check_threshold(threshold, data, percentile_func, name='threshold'):
    """ Checks if the given threshold is in correct format and within the limit.

    If necessary, this function also returns score of the data calculated based
    upon the given specific percentile function.
    Note: This is only for threshold as string.

    Parameters
    ----------
    threshold: float or str
        If threshold is a float value, it should be within the range of the
        maximum intensity value of the data.
        If threshold is a percentage expressed in a string it must finish with a
        percent sign like "99.7%".
    data: ndarray
        an array of the input masked data.
    percentile_func: function {scoreatpercentile, fastabspercentile}
        Percentile function for example scipy.stats.scoreatpercentile
        to calculate the score on the data.
    name: str, optional
        A string just used for representing the name of the threshold for a precise
        error message.

    Returns
    -------
    threshold: number
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

        threshold = percentile_func(data, percentile)
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


def check_parameters_megatrawls_datasets(inputs, standards, name):
    """ Checks given inputs against standards for megatrawls datasets parameters.

    If parameters are valid, then no error message is raised else error message
    will be raised.

    Parameters
    ----------
    inputs: list of str
        list to be checked.
    standards: list of str
        The given inputs will be checked against this given standards.
    name: str
        Used for precise naming in error message
    """
    message = ("Invalid {0} name is given: {1}. "
               "Please choose either of them {2}")

    if isinstance(inputs, _basestring):
        raise TypeError("Input given for {0} should be in list. "
                        "You have given as single variable: {1}".format(name, inputs))
    else:
        for each_input in inputs:
            if each_input not in standards:
                raise ValueError(message.format(name, each_input, str(standards)))
    return inputs
