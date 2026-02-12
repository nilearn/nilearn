"""Utility functions for the benchmarks."""

from typing import Literal

import nibabel as nib
from nibabel import Nifti1Image

from nilearn.image import load_img


def load(
    loader: Literal["nilearn", "nibabel (ref)"],
    n_masks: int = 1,
    n_subjects: int = 10,
) -> tuple[list[Nifti1Image] | Nifti1Image, Nifti1Image]:
    """
    There are already some masks and an fMRI image in the cache directory
    created by the setup_cache method in the Benchmark class. This function
    loads as many masks and the selected fMRI image from there.

    Parameters
    ----------
    loader : str
        The loader to use. Can be either 'nilearn' or 'nibabel (ref)'. When
        'nilearn' is selected, the load_img function from nilearn.image is
        used. When 'nibabel (ref)' is selected, the load function from nibabel
        is used.
    n_masks : int, default=1
        The number of masks to load.
    n_subjects : int, default=10
        This parameter refers to the number of subjects images concatenated
        together to create the fMRI image in the cache directory. The fMRI
        image is named 'fmri_{n_subjects}.nii.gz'.
    """
    loader_to_func = {
        "nilearn": load_img,
        "nibabel (ref)": nib.load,
    }
    loading_func = loader_to_func[loader]
    if n_masks < 1:
        raise ValueError("Number of masks must be at least 1.")
    elif n_masks == 1:
        return loading_func("mask_1.nii.gz"), loading_func(
            f"fmri_{n_subjects}.nii.gz"
        )
    else:
        return [
            loading_func(f"mask_{idx}.nii.gz") for idx in range(1, n_masks + 1)
        ], loading_func(f"fmri_{n_subjects}.nii.gz")
