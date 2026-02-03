"""Fixtures specific for maskers."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_array_equal

from nilearn.conftest import _make_mesh
from nilearn.image import get_data
from nilearn.surface import SurfaceImage


@pytest.fixture
def data_1(shape_3d_default):
    """Return 3D zeros with a few 10 in the center."""
    data = np.zeros(shape_3d_default)
    data[2:-2, 2:-2, 2:-2] = 10
    return data


@pytest.fixture
def mask_img_1(data_1, affine_eye):
    """Return a mask image."""
    return Nifti1Image(data_1.astype("uint8"), affine_eye)


@pytest.fixture
def shape_mask():
    """Shape for masks."""
    return (13, 14, 15)


def sklearn_surf_label_img() -> SurfaceImage:
    """Create a sample surface label image using the sample mesh,
    just to use for scikit-learn checks.
    """
    labels = {
        "left": np.asarray([1, 1, 2, 2]),
        "right": np.asarray([1, 1, 2, 2, 2]),
    }
    return SurfaceImage(_make_mesh(), labels)


def check_nifti_labels_masker_post_fit(
    masker,
    expected_n_regions: int,
    ref_shape: tuple[int, int, int] | None = None,
    ref_affine=None,
) -> None:
    """Run some common check on NiftiLabelsMasker post fit."""
    if ref_affine is not None:
        assert_array_equal(masker.labels_img_.affine, ref_affine)
        if masker.mask_img_ is not None:
            assert_array_equal(masker.mask_img_.affine, ref_affine)

    if ref_shape:
        assert len(ref_shape) == 3, "len(ref_shape) must be 3"
        assert masker.labels_img_.shape == ref_shape
        if masker.mask_img_ is not None:
            assert masker.mask_img_.shape == ref_shape

    assert masker.n_elements_ == expected_n_regions

    resampled_labels_img = masker.labels_img_
    labels = np.unique(get_data(resampled_labels_img))
    len(labels)

    # if masker.background_label in labels:
    #     assert n_resampled_labels == expected_n_regions + 1
    # else:
    #     assert n_resampled_labels == expected_n_regions

    # if hasattr(masker, "_lut_"):
    #     # get the LUT that tracks content after masking / resampling
    #     # done at transform time
    #     # if it exists
    #     assert len(masker._lut_) == n_resampled_labels
    # else:
    #     assert len(masker.lut_) == n_resampled_labels


def check_nifti_labels_masker_post_transform(
    masker,
    expected_n_regions: int,
    signals: np.ndarray,
    length: int | None = None,
    ref_shape: tuple[int, int, int] | None = None,
    ref_affine=None,
) -> None:
    """Run some common check on NiftiLabelsMasker post transform."""
    check_nifti_labels_masker_post_fit(
        masker, expected_n_regions, ref_shape, ref_affine
    )

    _check_signals(
        masker, expected_n_regions, signals, length, ref_affine, ref_shape
    )


def check_nifti_maps_masker_post_fit(
    masker, expected_n_regions: int, ref_shape=None, ref_affine=None
) -> None:
    """Run some common check on NiftiMapsMasker post fit."""
    assert masker.n_elements_ == expected_n_regions

    if ref_affine is not None:
        assert_array_equal(masker.maps_img_.affine, ref_affine)
        if masker.mask_img_ is not None:
            assert_almost_equal(masker.mask_img_.affine, ref_affine)

    if ref_shape:
        assert len(ref_shape) == 3, "len(ref_shape) must be 3"
        assert masker.maps_img_.shape[:3] == ref_shape
        if masker.mask_img_ is not None:
            assert masker.mask_img_.shape == ref_shape


def check_nifti_maps_masker_post_transform(
    masker,
    expected_n_regions: int,
    signals: np.ndarray,
    length: int | None = None,
    ref_shape: tuple[int, int, int] | None = None,
    ref_affine=None,
) -> None:
    """Run some common check on NiftiMapsMasker post transform.

    - check shape of signal
    - ensure that signal can be turned back into an image
      with expected shapre and affine
    """
    check_nifti_maps_masker_post_fit(
        masker, expected_n_regions, ref_shape, ref_affine
    )

    _check_signals(
        masker, expected_n_regions, signals, length, ref_affine, ref_shape
    )


def _check_signals(
    masker,
    expected_n_regions: int,
    signals,
    length: int | None = None,
    ref_affine=None,
    ref_shape: tuple[int, int, int] | None = None,
) -> None:
    """Run check on signals obtained from transform.

    - check shape of signal
    - ensure that signal can be turned back into an image
      with expected shapre and affine
    """
    if not isinstance(signals, list):
        signals = [signals]

    for s in signals:
        if length is None:
            assert s.shape[-1] == expected_n_regions
        else:
            assert s.shape == (length, expected_n_regions)

        img = masker.inverse_transform(s)

        if ref_affine is not None:
            assert_array_equal(img.affine, ref_affine)
        if ref_shape:
            assert img.shape[:3] == ref_shape[:3]
        if length:
            assert img.shape[-1] == length
