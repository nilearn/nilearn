"""Fixtures specific for maskers."""

import numpy as np
import pytest
from nibabel import Nifti1Image

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


def check_nifti_labels_masker_post_fit(masker, expected_n_regions):
    """Run some common check on NiftiLabelsMasker post fit."""
    assert masker.n_elements_ == expected_n_regions

    resampled_labels_img = masker.labels_img_
    n_resampled_labels = len(np.unique(get_data(resampled_labels_img)))

    assert n_resampled_labels == expected_n_regions + 1


def check_nifti_labels_masker_post_transform(
    masker, expected_n_regions, signals, length=None
):
    """Run some common check on NiftiLabelsMasker post transform.

    - check shape of signal
    - ensure that signal can be turned back into an image
    """
    check_nifti_labels_masker_post_fit(masker, expected_n_regions)

    if not isinstance(signals, list):
        signals = [signals]

    for s in signals:
        if length is None:
            assert s.shape[1] == expected_n_regions
        else:
            assert s.shape == (length, expected_n_regions)

        masker.inverse_transform(s)


def check_nifti_maps_masker_post_fit(masker, expected_n_regions):
    """Run some common check on NiftiMapsMasker post fit."""
    assert masker.n_elements_ == expected_n_regions


def check_nifti_maps_masker_post_transform(
    masker, expected_n_regions, signals, length=None
):
    """Run some common check on NiftiMapsMasker post transform.

    - check shape of signal
    - ensure that signal can be turned back into an image
    """
    assert masker.n_elements_ == expected_n_regions

    if not isinstance(signals, list):
        signals = [signals]

    for s in signals:
        if length is None:
            assert s.shape[-1] == expected_n_regions
        else:
            assert s.shape == (length, expected_n_regions)

        masker.inverse_transform(s)
