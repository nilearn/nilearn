"""Common utilities for the benchmarks."""

import nibabel as nib

from nilearn.datasets import fetch_adhd, fetch_atlas_basc_multiscale_2015
from nilearn.image import concat_imgs, load_img, new_img_like, resample_to_img


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
        atlas_path = fetch_atlas_basc_multiscale_2015(resolution=64).maps
        resampled_atlas = resample_to_img(
            atlas_path,
            concat,
            interpolation="nearest",
            copy_header=True,
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
