
from ._utils import _deprecation_warning

from nilearn.maskers.base_masker import *  # noqa

deprecated_path = 'nilearn.input_data.base_masker'
correct_path = 'nilearn.maskers.base_masker'

_deprecation_warning(deprecated_path, correct_path)

