
from ._utils import _deprecation_warning

from nilearn.maskers.nifti_labels_masker import *  # noqa

deprecated_path = 'nilearn.input_data.nifti_labels_masker'
correct_path = 'nilearn.maskers.nifti_labels_masker'

_deprecation_warning(deprecated_path, correct_path)

