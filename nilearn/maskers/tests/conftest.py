"""Fixtures specific for maskers."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal

from nilearn.conftest import _make_mesh
from nilearn.maskers import NiftiMapsMasker
from nilearn.surface import SurfaceImage


def check_signal_and_inverse_transform(
    masker,
    signal: np.ndarray | list[np.ndarray],
    expected_length: int | None = None,
) -> None:
    """Check signal extracted by a masker is
    inverse transformable by that same masker.
    """
    # multimaskers return signals as list
    # so let's treat all signals as list
    signal_list: list[np.ndarray] = (
        [signal] if not isinstance(signal, list) else signal
    )

    for s in signal_list:
        if expected_length is not None:
            assert s.shape == (expected_length, masker.n_elements_)
        else:
            assert s.shape[1] == masker.n_elements_

        img = masker.inverse_transform(s)

        # TODO adapt for other maskers
        if isinstance(masker, NiftiMapsMasker):
            ref_img = masker.maps_img_
            assert_almost_equal(img.affine, ref_img.affine)
            assert img.shape == ((*ref_img.shape[:3], s.shape[0]))


@pytest.fixture
def data_1(shape_3d_default) -> np.ndarray:
    """Return 3D zeros with a few 10 in the center."""
    data = np.zeros(shape_3d_default)
    data[2:-2, 2:-2, 2:-2] = 10
    return data


@pytest.fixture
def mask_img_1(data_1, affine_eye) -> Nifti1Image:
    """Return a mask image."""
    return Nifti1Image(data_1.astype("uint8"), affine_eye)


@pytest.fixture
def shape_mask() -> tuple[int, int, int]:
    """Shape for masks."""
    return (13, 14, 15)


def sklearn_surf_label_img(n_regions: int = 2) -> SurfaceImage:
    """Create a sample surface label image using the sample mesh,
    just to use for scikit-learn and nilearn checks.
    """
    if n_regions not in [1, 2]:
        raise ValueError(
            f"'n_regions' must be one of [1, 2]. Got {n_regions=}"
        )

    labels = {
        "left": np.asarray([1, 1, 1, 1]),
        "right": np.asarray([2, 2, 2, 1, 1]),
    }
    labels["left"][labels["left"] > n_regions] = 0
    labels["right"][labels["right"] > n_regions] = 0

    return SurfaceImage(_make_mesh(), labels)
