"""Old interface module to bids glm.

TODO: (nilearn >=0.15.0) remove
"""

import warnings
from typing import TYPE_CHECKING

from nilearn._utils.logger import find_stack_level

if TYPE_CHECKING:
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.glm.second_level import SecondLevelModel


def save_glm_to_bids(*args, **kwgars) -> "FirstLevelModel | SecondLevelModel":
    """Redirect to save_glm_to_bids from nilearn.glm."""
    from nilearn.glm.io import save_glm_to_bids as sgtb

    warnings.warn(
        (
            "'save_glm_to_bids' has been moved "
            "to 'nilearn.glm' in version 0.13.\n"
            "Importing from 'nilearn.interfaces' will be possible "
            "until Nilearn 0.15.0.\n"
            "Please import from 'nilearn.glm' instead."
        ),
        category=FutureWarning,
        stacklevel=find_stack_level(),
    )
    return sgtb(*args, **kwgars)
