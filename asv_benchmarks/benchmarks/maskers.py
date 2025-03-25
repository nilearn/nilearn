"""Benchmarks for masker objects under nilearn.maskers module."""

# ruff: noqa: RUF012

import numpy as np

from nilearn.maskers import NiftiMasker

from .common import Benchmark, load


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


class NiftiMaskingVsReference(Benchmark):
    """
    Comparison between the performance of applying a mask to an image using
    nilearn vs. numpy.
    """

    # here we vary both the implementation and the loader
    # so masking can be done using nilearn or numpy (implementation)
    # and the mask and image can be loaded using nilearn or nibabel (loader)
    param_names = ["implementation", "loader"]
    params = (["nilearn", "numpy (ref)"], ["nilearn", "nibabel (ref)"])

    def time_masker(self, implementation, loader):
        """Time the loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)

    def peakmem_masker(self, implementation, loader):
        """Peak memory of loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)


class NiftiMasking(Benchmark):
    """
    Benchmark for applying a mask to an image using nilearn with different
    parameters.
    """

    # try different combinations of parameters for the NiftiMasker object
    param_names = ["smoothing_fwhm", "standardize", "detrend"]
    params = (
        [None, 6],
        [False, "zscore_sample", "zscore", "psc"],
        [False, True],
    )

    def time_masker(
        self,
        smoothing_fwhm,
        standardize,
        detrend,
    ):
        """Time the loading (only with nilearn here) and then masking with
        different parameters.
        """
        mask, img = load("nilearn")
        apply_mask(
            mask,
            img,
            "nilearn",
            nifti_masker_params={
                "smoothing_fwhm": smoothing_fwhm,
                "standardize": standardize,
                "detrend": detrend,
            },
        )

    def peakmem_masker(
        self,
        smoothing_fwhm,
        standardize,
        detrend,
    ):
        """Peak memory for the loading (only with nilearn here) and then
        masking with different parameters.
        """
        mask, img = load("nilearn")
        apply_mask(
            mask,
            img,
            "nilearn",
            nifti_masker_params={
                "smoothing_fwhm": smoothing_fwhm,
                "standardize": standardize,
                "detrend": detrend,
            },
        )
