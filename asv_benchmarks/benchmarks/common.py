"""Common Benchmarks class that does the setup for the benchmarks."""

from nilearn.datasets import fetch_adhd, fetch_atlas_schaefer_2018
from nilearn.image import concat_imgs, new_img_like, resample_to_img


class Benchmark:
    """
    Base class for the benchmarks. Currently, it only contains a method to
    setup the cache which is used to store the images and masks used in the
    benchmarks.
    """

    def setup_cache(self, n_subjects=10, n_masks=1):
        """Set up the cache directory with the necessary images and masks.

        The fMRI image is created by concatenating n_subjects subject images
        from :func:`nilearn.datasets.fetch_adhd`. The masks are created by
        resampling the atlas from
        :func:`nilearn.datasets.fetch_atlas_basc_multiscale_2015` to the fMRI
        image and then creating masks for each region in the atlas.

        Parameters
        ----------
        n_subjects : int, optional, default=10
            The number of subject images concatenated together to create the
            fMRI image.
        n_masks : int, optional, default=1
            The number of masks to create.
        """
        # get an image
        fmri_data = fetch_adhd(n_subjects=n_subjects)
        concat = concat_imgs(fmri_data.func)
        concat.to_filename(f"fmri_{n_subjects}.nii.gz")

        # get a mask
        atlas_path = fetch_atlas_schaefer_2018(n_rois=100).maps
        resampled_atlas = resample_to_img(
            atlas_path,
            concat,
            interpolation="nearest",
            force_resample=True,
        )
        for idx in range(1, n_masks + 1):
            mask = resampled_atlas.get_fdata() == idx
            mask_img = new_img_like(
                resampled_atlas,
                mask,
                affine=resampled_atlas.affine,
                copy_header=True,
            )
            mask_img.to_filename(f"mask_{idx}.nii.gz")
