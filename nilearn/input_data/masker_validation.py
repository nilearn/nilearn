
from ._utils import _deprecation_warning

from nilearn.maskers.masker_validation import *  # noqa

deprecated_path = 'nilearn.input_data.masker_validation'
correct_path = 'nilearn.maskers.masker_validation'

_deprecation_warning(deprecated_path, correct_path)

