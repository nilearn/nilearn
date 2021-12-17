
from ._utils import _deprecation_warning

from nilearn.maskers.nifti_spheres_masker import *  # noqa

deprecated_path = 'nilearn.input_data.nifti_sphere_masker'
correct_path = 'nilearn.maskers.nifti_sphere_masker'

_deprecation_warning(deprecated_path, correct_path)

