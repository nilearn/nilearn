"""
Machine Learning module for NeuroImaging in python
==================================================

NiLearn aims to simplify the use of the scikit-learn in the context of
neuroimaging. It provides specific input/output functions, algorithms and
visualisation tools.

See http://nilearn.github.com for complete documentation.
"""

__version__ = "0.1a"

def _check_dependencies():
    try:
        import numpy
    except ImportError:
        print ('Numpy could not be found,'
        ' please install it properly to use nilearn.')

    try:
        import scipy
    except ImportError:
        print ('Scipy could not be found,'
               ' please install it properly to use nilearn.')

    try:
        import sklearn
    except ImportError:
        print ('Scikit-learn could not be found,'
               ' please install it properly to use nilearn.')

_check_dependencies()

