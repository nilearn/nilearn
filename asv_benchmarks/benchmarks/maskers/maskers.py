"""Benchmarks for masker objects under nilearn.maskers module."""

from ..common import Benchmark
from ..utils import apply_mask, load


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

    def time_nifti_masker(self, smoothing_fwhm, detrend):
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

    def peakmem_nifti_masker(self, smoothing_fwhm, detrend):
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

    def time_compare_mask(self, implementation, loader):
        """Time the loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)

    def peakmem_compare_mask(self, implementation, loader):
        """Peak memory of loading and then masking."""
        mask, img = load(loader)
        apply_mask(mask, img, implementation)
