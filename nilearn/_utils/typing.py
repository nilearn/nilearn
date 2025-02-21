from pathlib import Path

from nibabel import Nifti1Image

NiimgLike = (Nifti1Image, str, Path)
