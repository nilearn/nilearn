"""
Machine Learning module for NeuroImaging in python
==================================================

Nisl aims to simplify the use of the scikit-learn in the contexte of
neuroimaging. It provides specific input/output function, algorithms and
visualisation tools.

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
    print ('Scikit-learn could not be found,'
           ' please install it properly to use nisl.')

__version__ = 0.1
