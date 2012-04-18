"""
Machine Learning module for NeuroImaging in python
==================================================

See http://nisl.github.com for complete documentation.
"""

try:
    import numpy
except ImportError:
    print 'Numpy could not be found, please install it properly to use nisl.'


try:
    import scipy
except ImportError:
    print 'Scipy could not be found, please install it properly to use nisl.'

try:
    import sklearn
except ImportError:
    print 'Sklearn could not be found, please install it properly to use nisl.'


try:
    from numpy.testing import nosetester

    class NoseTester(nosetester.NoseTester):
        """ Subclass numpy's NoseTester to add doctests by default
        """

        def test(self, label='fast', verbose=1, extra_argv=['--exe'],
                        doctests=True, coverage=False):
            """Run the full test suite

            Examples
            --------
            This will run the test suite and stop at the first failing
            example
            >>> from nisl import test
            >>> test(extra_argv=['--exe', '-sx']) #doctest: +SKIP
            """
            return super(NoseTester, self).test(label=label, verbose=verbose,
                                    extra_argv=extra_argv,
                                    doctests=doctests, coverage=coverage)

    test = NoseTester().test
    del nosetester
except:
    pass


__all__ = ['datasets']

__version__ = '2010'
