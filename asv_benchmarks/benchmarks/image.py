"""Benchmarks for image operations under nilearn.image module."""

# ruff: noqa: RUF012

from nilearn.image import mean_img

from .common import Benchmark, load


class Loading(Benchmark):
    """
    A benchmark that measures the performance of loading images from
    disk using nibabel and nilearn.
    """

    # compare loading images using nibabel and nilearn
    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_loading(self, loader):
        """Time the loading of images."""
        load(loader)

    def peakmem_loading(self, loader):
        """Peak memory of loading images."""
        load(loader)


class Mean(Benchmark):
    """
    An example benchmark that measures the performance of computing the mean
    of a 4D image using nibabel and nilearn.
    """

    # compare loading images using nibabel and nilearn
    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_mean(self, loader):
        """Time the loading followed by taking the mean."""
        img = load(loader)[1]
        mean_img(img, copy_header=True)

    def peakmem_mean(self, loader):
        """Peak memory of loading followed by taking the mean."""
        img = load(loader)[1]
        mean_img(img, copy_header=True)


class Slicing(Benchmark):
    """
    An example benchmark that measures the performance of slicing a 4D image
    using nibabel and nilearn.
    """

    # compare loading images using nibabel and nilearn
    param_names = ["loader"]
    params = ["nilearn", "nibabel (ref)"]

    def time_slicing(self, loader):
        """Time the loading the image followed by extracting a slice of it."""
        img = load(loader)[1]
        img.dataobj[..., 0]

    def peakmem_slicing(self, loader):
        """Peak memory of loading the image followed by extracting a
        slice of it.
        """
        img = load(loader)[1]
        img.dataobj[..., 0]
