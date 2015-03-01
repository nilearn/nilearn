"""
Machine Learning module for NeuroImaging in python
==================================================

Nilearn aims to simplify the use of the scikit-learn in the context of
neuroimaging. It provides specific input/output functions, algorithms and
visualization tools.

See http://nilearn.github.io for complete documentation.
"""
import gzip
from distutils.version import LooseVersion

from .version import _check_module_dependencies, get_min_version, __version__


_check_module_dependencies()

# Monkey-patch gzip to have faster reads on large gzip files
if hasattr(gzip.GzipFile, 'max_read_chunk'):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb

# Boolean controling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
check_cache_version = True

# These are warnings that we may trigger, but that we can do nothing about.
# Hide them from users; warnings should be triggered for things that they
# can act on only!
import warnings

# We know this, no need to expose our users to it.
warnings.filterwarnings('ignore', 'check_cv will return indices instead of boolean masks', DeprecationWarning)
warnings.filterwarnings('ignore', 'The compiler package is deprecated and removed in Python 3.x.', DeprecationWarning)

# WardAgglomeration cannot be migrated to AgglomerativeClustering until
#   our scikit-learn minv_version is 0.15.
if LooseVersion(get_min_version('sklearn')) < LooseVersion('0.15'):
    warnings.filterwarnings('once', 'The Ward class is deprecated since 0.14 and will be removed in 0.17.')
