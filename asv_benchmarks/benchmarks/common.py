"""Common Benchmarks class that does the setup for the benchmarks."""

import numpy as np
from nibabel import Nifti1Image
from scipy.signal import get_window


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

    DO NOT CHANGE n_regions (some tests expect this value).
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


class Benchmark:
    """
    Base class for the benchmarks. Currently, it only contains a method to
    setup the cache which is used to store the images and masks used in the
    benchmarks.
    """

    timeout = 2400  # 40 mins

    def setup_cache(self, n_subjects: int = 10, n_masks: int = 1):
        """Set up the cache directory with the necessary images and masks.

        The fMRI image is created by concatenating n_subjects subject images
        from :func:`nilearn.datasets.fetch_abide_pcp`. The masks are created by
        resampling the atlas from
        :func:`nilearn.datasets.fetch_atlas_schaefer_2018` to the fMRI
        image and then creating masks for each region in the atlas.

        Parameters
        ----------
        n_subjects : int, default=10
            The number of subject images concatenated
            together to create the fMRI image.

        n_masks : int, default=1
            The number of masks to create.
        """
        n_vol_per_subject = 190
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
            length=n_vol_per_subject * n_subjects,
            affine=affine,
        )
        img.to_filename(f"fmri_{n_subjects}.nii.gz")

        # create a mask
        atlas = img_labels(shape=shape, affine=affine, n_regions=100)
        for idx in range(1, n_masks + 1):
            mask = atlas.get_fdata() == idx
            mask_img = Nifti1Image(
                mask.astype(np.int32),
                affine=atlas.affine,
            )
            mask_img.to_filename(f"mask_{idx}.nii.gz")
