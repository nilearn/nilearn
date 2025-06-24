"""Benchmarks for image operations under nilearn.image module."""

from nilearn.image import mean_img

from ..common import Benchmark
from ..utils import load


class LoadImgBenchmark(Benchmark):
    """Benchmark that measures the performance of loading images via
    nilearn.load_img.
    """

    def time_load_img(self):
        """Time the loading of images."""
        load("nilearn")

    def peakmem_load_img(self):
        """Peak memory of loading images."""
        load("nilearn")


class MeanImgBenchmark(Benchmark):
    """
    Benchmark that measures the performance of first loading and then
    computing the mean of a 4D image.
    """

    def time_mean_img(self):
        """Time the loading followed by taking the mean."""
        img = load("nilearn")[1]
        mean_img(img)

    def peakmem_mean_img(self):
        """Peak memory of loading followed by taking the mean."""
        img = load("nilearn")[1]
        mean_img(img)
