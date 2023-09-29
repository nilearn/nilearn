"""The :mod:`nilearn.experimental` module provides importable modules that \
enable the use of experimental features.

The features and estimators that are experimental aren't subject to
deprecation cycles. Use them at your own risks!
"""

import warnings

warnings.warn(
    "\n\n\nAll features in the nilearn.experimental module are "
    "experimental and subject to change. They are included "
    "in the nilearn package to gather early feedback from users "
    "about prototypes of new features. "
    "Changes may break backwards compatibility without prior "
    "notice or a deprecation cycle. Moreover, some features may "
    "be incomplete or may have been tested less thoroughly than "
    "the rest of the library.\n\n"
)

__all__ = ["surface"]
