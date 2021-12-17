
from ._utils import _deprecation_warning

from nilearn.maskers.multi_nifti_masker import *  # noqa

deprecated_path = 'nilearn.input_data.multi_nifti_masker'
correct_path = 'nilearn.maskers.multi_nifti_masker'

_deprecation_warning(deprecated_path, correct_path)

