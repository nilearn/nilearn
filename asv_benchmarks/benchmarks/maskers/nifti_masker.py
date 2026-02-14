"""Benchmarks for masker objects under nilearn.maskers module."""

from typing import Any, Literal

import numpy as np
from nibabel import Nifti1Image

from nilearn.maskers import NiftiMasker

from ..utils import Benchmark, load


def apply_mask(
    mask: Nifti1Image,
    img: Nifti1Image,
    implementation: Literal["nilearn", "numpy"],
    nifti_masker_params: None | dict[str, Any] = None,
):
    """Apply a mask to an image using nilearn or numpy.

    Parameters
    ----------
    mask : Nifti1Image
        The mask to apply.
    img : Nifti1Image
        The image to apply the mask to.
    implementation : str
        The implementation to use. Can be either 'nilearn' or 'numpy'.
    nifti_masker_params : dict, default=None
        Parameters to pass to the NiftiMasker object when using 'nilearn' as
        the implementation.
    """
    if implementation == "nilearn":
        masker = NiftiMasker(mask_img=mask)
        if nifti_masker_params is not None:
            masker.set_params(**nifti_masker_params)
        masker.fit_transform(img)
    elif implementation == "numpy":
        mask_data = np.asarray(mask.dataobj).astype(bool)
        img_data = np.asarray(img.dataobj)
        img_data[mask_data]


class NiftiMaskerBenchmark(Benchmark):
    """
    Benchmark for applying a mask to an image using NiftiMasker with different
    parameters.
    """

    # try different combinations of parameters for the NiftiMasker object
    param_names = ("smoothing_fwhm", "detrend")
    params = (
        [None, 6],
        [False, True],
    )

    def time_nifti_masker_fit_transform(self, smoothing_fwhm, detrend):
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
                "detrend": detrend,
            },
        )

    def peakmem_nifti_masker_fit_transform(self, smoothing_fwhm, detrend):
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
                "detrend": detrend,
            },
        )


class CompareMask(Benchmark):
    """
    Comparison between the performance of applying a mask to an image using
    nilearn vs. numpy.
    """

    # here we vary both the implementation and the loader
    # so masking can be done using nilearn or numpy (implementation)
    # and the mask and image can be loaded using nilearn or nibabel (loader)
    param_names = ("implementation", "loader")
    params = (["nilearn", "numpy (ref)"], ["nilearn", "nibabel (ref)"])

    def time_nifti_masker_fit_transform_compare(self, implementation, loader):
        """Time the loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)

    def peakmem_nifti_masker_fit_transform_compare(
        self, implementation, loader
    ):
        """Peak memory of loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)
