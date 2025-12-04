"""Module to be deprecated in nilearn 0.14.0.

Kept here for backward compatibility.
"""

import warnings

from nilearn._utils.logger import find_stack_level

MSG = (
    "The function '{}' was moved to 'nilearn.image'.\n"
    "Importing it from 'nilearn._utils.niimg_conversions' "
    "will remain possible till Nilearn 0.14.0.\n"
    "Please change to import from 'nilearn.image'."
)


def check_niimg(*args, **kwargs):
    from nilearn.image import check_niimg as cn

    warnings.warn(
        MSG.format("check_niimg"),
        category=DeprecationWarning,
        stacklevel=find_stack_level(),
    )
    return cn(*args, **kwargs)


def check_niimg_3d(*args, **kwargs):
    from nilearn.image import check_niimg_3d as cn3d

    warnings.warn(
        MSG.format("check_niimg_3d"),
        category=DeprecationWarning,
        stacklevel=find_stack_level(),
    )

    return cn3d(*args, **kwargs)


def check_niimg_4d(*args, **kwargs):
    from nilearn.image import check_niimg_4d as cn4d

    warnings.warn(
        MSG.format("check_niimg_4d"),
        category=DeprecationWarning,
        stacklevel=find_stack_level(),
    )

    return cn4d(*args, **kwargs)
