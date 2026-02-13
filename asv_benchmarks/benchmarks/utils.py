"""Common utils for all benchmarks."""

import contextlib
from typing import Literal

import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from scipy.signal import get_window

from nilearn.image import load_img

with contextlib.suppress(ImportError):
    pass

LENGTH = 10


def _rng():
    return np.random.default_rng(42)


def generate_fake_fmri(
    shape: tuple[int, int, int],
    affine,
    length: int,
    kind="noise",
) -> tuple[Nifti1Image, Nifti1Image]:
    """Generate a signal which can be used for benchmarks.

    The return value is a 4D image, representing 3D volumes along time.
    Only the voxels in the center are non-zero,
    to mimic the presence of brain voxels in real signals.

    Adapted from nilearn._utils.data_gen.generate_fake_fmri

    Parameters
    ----------
    shape : :obj:`tuple`
        Shape of 3D volume.

    affine : :obj:`numpy.ndarray`
        Affine of returned images.

    length : :obj:`int`
        Number of time instants.

    kind : :obj:`str`, default='noise'
        Kind of signal used as timeseries.
        "noise": uniformly sampled values in [0..255]
        "step": 0.5 for the first half then 1.

    Returns
    -------
    Niimg-like object
        Fake fmri signal.
        shape: shape + (length,)

    Niimg-like object
        Mask giving non-zero voxels.
    """
    full_shape = (*shape, length)
    fmri = np.zeros(full_shape)
    # Fill central voxels timeseries with random signals
    width = [s // 2 for s in shape]
    shift = [s // 4 for s in shape]

    rng = _rng()
    if kind == "noise":
        signals = rng.integers(256, size=([*width, length]))
    elif kind == "step":
        signals = np.ones([*width, length])
        signals[..., : length // 2] = 0.5
    else:
        raise ValueError("Unhandled value for parameter 'kind'")

    fmri[
        shift[0] : shift[0] + width[0],
        shift[1] : shift[1] + width[1],
        shift[2] : shift[2] + width[2],
        :,
    ] = signals

    mask = np.zeros(shape)
    mask[
        shift[0] : shift[0] + width[0],
        shift[1] : shift[1] + width[1],
        shift[2] : shift[2] + width[2],
    ] = 1

    return (Nifti1Image(fmri, affine), Nifti1Image(mask, affine))


def generate_regions_ts(n_features: int, n_regions: int):
    """Generate some regions as timeseries.

    adapted from nilearn._utils.data_gen.generate_regions_ts

    Parameters
    ----------
    n_features : :obj:`int`
        Number of features.

    n_regions : :obj:`int`
        Number of regions.

    Returns
    -------
    regions : :obj:`numpy.ndarray`
        Regions, represented as signals.
        shape (n_features, n_regions)

    """
    rng = _rng()
    window = "boxcar"
    overlap = 0

    assert n_features > n_regions

    # Compute region boundaries indices.
    # Start at 1 to avoid getting an empty region
    boundaries = np.zeros(n_regions + 1)
    boundaries[-1] = n_features
    boundaries[1:-1] = rng.permutation(np.arange(1, n_features))[
        : n_regions - 1
    ]
    boundaries.sort()

    regions = np.zeros((n_regions, n_features), order="C")
    overlap_end = int((overlap + 1) / 2.0)
    overlap_start = int(overlap / 2.0)
    for n in range(len(boundaries) - 1):
        start = int(max(0, boundaries[n] - overlap_start))
        end = int(min(n_features, boundaries[n + 1] + overlap_end))
        win = get_window(window, end - start)
        win /= win.mean()  # unity mean
        regions[n, start:end] = win

    return regions


def img_labels(
    shape: tuple[int, int, int], affine, n_regions: int
) -> Nifti1Image:
    """Generate fixture for default label image.

    adapted from nilearn._utils.data_gen.generate_labeled_regions

    """
    n_voxels = shape[0] * shape[1] * shape[2]

    n_regions += 1
    labels = range(n_regions)

    regions = generate_regions_ts(n_voxels, n_regions)
    # replace weights with labels
    for n, row in zip(labels, regions, strict=False):
        row[row > 0] = n
    data = np.zeros(shape, dtype="int32")
    data[np.ones(shape, dtype=bool)] = regions.sum(axis=0).T

    return Nifti1Image(data, affine)


def load(
    loader: Literal["nilearn", "nibabel (ref)"], n_masks: int = 1
) -> tuple[list[Nifti1Image] | Nifti1Image, Nifti1Image]:
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
    n_masks : int, default=1
        The number of masks to load.
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
            f"fmri_{LENGTH}.nii.gz"
        )
    else:
        return [
            loading_func(f"mask_{idx}.nii.gz") for idx in range(1, n_masks + 1)
        ], loading_func(f"fmri_{LENGTH}.nii.gz")


class Benchmark:
    """
    Base class for the benchmarks.

    Currently, it only contains a method to
    setup the cache which is used to store the images and masks used in the
    benchmarks.
    """

    timeout: int = 2400  # 40 mins
    length: int = LENGTH

    @property
    def fmri_filename(self):
        """Return name of fmri image used in benchmarks."""
        return f"fmri_{self.length}.nii.gz"

    def setup_cache(self, n_masks: int = 1):
        """Set up the cache directory with the necessary images and masks.

        The fMRI image is created by concatenating n_subjects subject images
        from :func:`nilearn.datasets.fetch_abide_pcp`. The masks are created by
        resampling the atlas from
        :func:`nilearn.datasets.fetch_atlas_schaefer_2018` to the fMRI
        image and then creating masks for each region in the atlas.

        Parameters
        ----------
        n_masks : int, default=1
            The number of masks to create.
        """
        shape = (60, 65, 70)

        affine = np.asarray(
            [
                [-3.0, -0.0, 0.0, 90.0],
                [-0.0, 3.0, -0.0, -126.0],
                [0.0, 0.0, 3.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        img, mask = generate_fake_fmri(
            shape=shape,
            length=self.length,
            affine=affine,
        )
        img.to_filename(self.fmri_filename)

        # create a mask
        atlas = img_labels(shape=shape, affine=affine, n_regions=100)
        for idx in range(1, n_masks + 1):
            mask = atlas.get_fdata() == idx
            mask_img = Nifti1Image(
                mask.astype(np.int32),
                affine=atlas.affine,
            )
            mask_img.to_filename(f"mask_{idx}.nii.gz")
