"""
Machine Learning module for NeuroImaging in python
==================================================

NiLearn aims to simplify the use of the scikit-learn in the context of
neuroimaging. It provides specific input/output functions, algorithms and
visualization tools.

See http://nilearn.github.com for complete documentation.
"""

from .version import _check_module_dependencies, __version__


# Each time nilearn is imported, check module dependencies
#   so that a user gets a useful error message immediately.
_check_module_dependencies()


# Monkey-patch gzip to have faster reads on large gzip files
#   Note: import must occur AFTER dependency checks occur above.
import gzip

if hasattr(gzip.GzipFile, 'max_read_chunk'):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb


# Boolean controling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
check_cache_version = True
