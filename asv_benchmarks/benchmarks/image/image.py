"""Benchmarks for image operations under nilearn.image module."""

from nilearn.image import index_img, mean_img

from ..utils import Benchmark, load


class LoadImgBenchmark(Benchmark):
    """Benchmark that measures the performance of loading images via
    nilearn.load_img.
    """

    def time_image_load_img(self):
        """Time the loading of images."""
        load("nilearn")

    def peakmem_image_load_img(self):
        """Peak memory of loading images."""
        load("nilearn")


class MeanImgBenchmark(Benchmark):
    """
    Benchmark that measures the performance of first loading and then
    computing the mean of a 4D image.
    """

    def time_image_mean_img(self):
        """Time the loading followed by taking the mean."""
        img = load("nilearn")[1]
        mean_img(img)

    def peakmem_image_mean_img(self):
        """Peak memory of loading followed by taking the mean."""
        img = load("nilearn")[1]
        mean_img(img)


class IndexImgBenchmark(Benchmark):
    """A benchmark that compares the performance of indexing a 4D image."""

    def time_image_index_img(self):
        """Time the loading the image followed by indexing a voxel."""
        index_img(
            self.fmri_filename,
            slice(1, self.n_vol_per_subject * self.n_subjects - 10, 2),
        )

    def peakmem_image_index_img(self):
        """Peak memory of loading the image followed by indexing a voxel."""
        index_img(
            self.fmri_filename,
            slice(1, self.n_vol_per_subject * self.n_subjects - 10, 2),
        )


class CompareLoad(Benchmark):
    """
    A benchmark that compares the performance of loading images from
    disk using nibabel and nilearn.
    """

    # compare loading images using nibabel and nilearn
    param_names = ("loader",)
    params = ("nilearn", "nibabel (ref)")

    def time_image_compare_load(self, loader):
        """Time the loading of images."""
        load(loader)

    def peakmem_image_compare_load(self, loader):
        """Peak memory of loading images."""
        load(loader)


class CompareMean(Benchmark):
    """
    A benchmark that compares the performance of first computing the mean
    of a 4D image loaded via nibabel and nilearn.
    """

    # compare loading images using nibabel and nilearn
    param_names = ("loader",)
    params = ("nilearn", "nibabel (ref)")

    def time_image_compare_mean_img(self, loader):
        """Time the loading followed by taking the mean."""
        img = load(loader)[1]
        mean_img(img)

    def peakmem_image_compare_mean_img(self, loader):
        """Peak memory of loading followed by taking the mean."""
        img = load(loader)[1]
        mean_img(img)


class CompareSlice(Benchmark):
    """
    A benchmark that compares the performance of slicing a 4D image loaded
    via nibabel and nilearn.
    """

    # compare loading images using nibabel and nilearn
    param_names = ("loader",)
    params = ("nilearn", "nibabel (ref)")

    def time_image_compare_slice(self, loader):
        """Time the loading the image followed by extracting a slice of it."""
        img = load(loader)[1]
        img.dataobj[..., 0]

    def peakmem_image_compare_slice(self, loader):
        """Peak memory of loading the image followed by extracting a
        slice of it.
        """
        img = load(loader)[1]
        img.dataobj[..., 0]
