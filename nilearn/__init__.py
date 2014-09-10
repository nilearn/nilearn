"""
Machine Learning module for NeuroImaging in python
==================================================

NiLearn aims to simplify the use of the scikit-learn in the context of
neuroimaging. It provides specific input/output functions, algorithms and
visualization tools.

See http://nilearn.github.com for complete documentation.
"""

__version__ = "0.1a"


def _import_module_with_version_check(module_name, minimum_version):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('{} could not be found, '
                              'please install it properly to use nilearn'
                              ).format(module_name)
        exc.args += (user_friendly_info,)
        raise

    module_version = module.__version__

    version_too_old = (not LooseVersion(module_version) >=
                       LooseVersion(minimum_version))

    if version_too_old:
        message = (
            'A {module_name} version of at least {minimum_version} '
            'is required to use nilearn. {module_version} was found. '
            'Please upgrade {module_name}').format(
                module_name=module_name,
                minimum_version=minimum_version,
                module_version=module_version)

        raise ImportError(message)

    return module


def _check_dependencies():
    _required_module_versions = [('numpy', '1.6.0'),
                                 ('scipy', '0.9.0'),
                                 ('sklearn', '0.10'),
                                 ('nibabel', '1.1.0')]

    for module_name, minimum_version in _required_module_versions:
        _import_module_with_version_check(module_name, minimum_version)

    try:
        import gzip
        if hasattr(gzip.GzipFile, 'max_read_chunk'):
            # Monkey-patch gzip to have faster reads on large
            # gzip files
            gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb
    except ImportError as exc:
        exc.args += ('Python has been compiled without gzip, '
                     'reading nii.gz files will be impossible.',)
        raise

_check_dependencies()

del _import_module_with_version_check
del _check_dependencies

# Boolean controling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
check_cache_version = True
