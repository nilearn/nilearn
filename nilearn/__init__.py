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
    from distutils.version import LooseVersion
    try:
        import numpy
    except ImportError:
        print('Numpy could not be found,'
              ' please install it properly to use nilearn.')
    if not LooseVersion(numpy.__version__) >= LooseVersion('1.6.0'):
        raise ImportError('A numpy version of at least 1.6 is required '
            'to use nilearn. %s was found. Please upgrade numpy.'
            % numpy.__version__
            )

    try:
        import scipy
    except ImportError:
        print('Scipy could not be found,'
              ' please install it properly to use nilearn.')
    if not LooseVersion(scipy.__version__) >= LooseVersion('0.9.0'):
        raise ImportError('A scipy version of at least 0.9 is required '
            'to use nilearn. %s was found. Please upgrade scipy.'
            % scipy.__version__
            )

    try:
        import sklearn
    except ImportError:
        print('Scikit-learn could not be found,'
              ' please install it properly to use nilearn.')
    if not LooseVersion(sklearn.__version__) >= LooseVersion('0.10'):
        raise ImportError('A scikit-learn version of at least 0.10 is required'
            ' to use nilearn. %s was found. Please upgrade scikit-learn.'
            % sklearn.__version__
            )


    try:
        import nibabel
    except ImportError:
        print('nibabel could not be found,'
              ' please install it properly to use nilearn.')
    try:
        import gzip
        if hasattr(gzip.GzipFile, 'max_read_chunk'):
            # Monkey-patch gzip to have faster reads on large
            # gzip files
            gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024 # 100Mb
    except ImportError:
        print('Python has been compiled without gzip,'
              ' reading nii.gz files will be impossible.')
    if not LooseVersion(nibabel.__version__) >= LooseVersion('1.1.0'):
        raise ImportError('A nibabel version of at least 1.1 is required'
            ' to use nilearn. %s was found. Please upgrade nibabel.'
            % nibabel.__version__
            )


_check_dependencies()


del _check_dependencies

# Boolean controling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
check_cache_version = True

