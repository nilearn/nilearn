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

    try:
        import nibabel
    except ImportError:
        print ('nibabel could not be found,'
               ' please install it properly to use nilearn.')
    try:
        import gzip
        if hasattr(gzip.GzipFile, 'max_read_chunk'):
            # Monkey-patch gzip to have faster reads on large
            # gzip files
            gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024 # 100Mb
    except ImportError:
        print ('Python has been compiled without gzip,'
               ' reading nii.gz files will be impossible.')


_check_dependencies()


del _check_dependencies

# Boolean controling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
check_cache_version = True

