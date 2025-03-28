"""Utility functions for the benchmarks."""

import nibabel as nib
import numpy as np

from nilearn.image import load_img
from nilearn.maskers import NiftiMasker


def load(loader, n_masks=1, n_subjects=10):
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
    n_masks : int, optional, default=1
        The number of masks to load.
    n_subjects : int, optional, default=10
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


def apply_mask(mask, img, implementation, nifti_masker_params=None):
    """Apply a mask to an image using nilearn or numpy.

    Parameters
    ----------
    mask : Nifti1Image
        The mask to apply.
    img : Nifti1Image
        The image to apply the mask to.
    implementation : str
        The implementation to use. Can be either 'nilearn' or 'numpy'.
    nifti_masker_params : dict, optional, default=None
        Parameters to pass to the NiftiMasker object when using 'nilearn' as
        the implementation.
    """
    if implementation == "nilearn":
        if nifti_masker_params is None:
            NiftiMasker(mask_img=mask).fit_transform(img)
        else:
            masker = NiftiMasker(mask_img=mask)
            masker.set_params(**nifti_masker_params)
            masker.fit_transform(img)
    elif implementation == "numpy":
        mask = np.asarray(mask.dataobj).astype(bool)
        img = np.asarray(img.dataobj)
        img[mask]
