"""Benchmarks for masker objects under nilearn.maskers module."""

from ..common import Benchmark
from ..utils import apply_mask, load


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
