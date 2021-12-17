
from ._utils import _deprecation_warning

from nilearn.maskers.nifti_maps_masker import *  # noqa

deprecated_path = 'nilearn.input_data.nifti_maps_masker'
correct_path = 'nilearn.maskers.nifti_maps_masker'

_deprecation_warning(deprecated_path, correct_path)

