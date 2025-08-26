"""Test CanICA."""

import sys

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from nilearn.decomposition.canica import CanICA
from nilearn.decomposition.tests.conftest import (
    RANDOM_STATE,
    check_decomposition_estimator,
)
from nilearn.image import get_data, iter_img
from nilearn.surface.surface import get_data as get_surface_data


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_threshold_bound_error(canica_data_single_img):
    """Test that an error is raised when the threshold is higher \
    than the number of components.
    """
    with pytest.raises(ValueError, match="Threshold must not be higher"):
        canica = CanICA(n_components=4, threshold=5.0, smoothing_fwhm=None)
        canica.fit(canica_data_single_img)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_percentile_range(rng, canica_data_single_img):
    """Test that a warning is given when thresholds are stressed."""
    edge_case = rng.integers(low=1, high=10)

    # stress thresholding via edge case
    canica = CanICA(
        n_components=edge_case,
        threshold=float(edge_case),
        smoothing_fwhm=None,
    )

    with pytest.warns(UserWarning, match="obtained a critical threshold"):
        canica.fit(canica_data_single_img)


# TODO (python >= 3.10) remove skipif when dropping python 3.9
@pytest.mark.skipif(
    sys.version_info[1] == 9,
    reason="fails only on MacOS with python 3.9",
)
@pytest.mark.parametrize("data_type", ["nifti"])
def test_canica_square_img(
    decomposition_mask_img, canica_components, canica_data
):
    """Check content of components."""
    # We do a large number of inits to be sure to find the good match

    # Note that
    # adding smoothing will make this test break
    smoothing_fwhm = None

    canica = CanICA(
        n_components=4,
        random_state=RANDOM_STATE,
        mask=decomposition_mask_img,
        smoothing_fwhm=smoothing_fwhm,
        n_init=50,
    )
    canica.fit(canica_data)
    maps = get_data(canica.components_img_)
    maps = np.rollaxis(maps, 3, 0)

    # FIXME: This could be done more efficiently, e.g. thanks to hungarian
    # Find pairs of matching components
    # compute the cross-correlation matrix between components
    mask = get_data(decomposition_mask_img) != 0
    K = np.corrcoef(canica_components[:, mask.ravel()], maps[:, mask])[4:, :4]

    # K should be a permutation matrix, hence its coefficients
    # should all be close to 0 1 or -1
    K_abs = np.abs(K)

    assert np.sum(K_abs > 0.9) == 4

    K_abs[K_abs > 0.9] -= 1

    assert_array_almost_equal(K_abs, 0, 1)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_component_sign(canica_data, data_type):
    """Check sign of extracted components.

    Regression test:
    We should have a heuristic that flips the sign of components in
    DictLearning to have more positive values than negative values, for
    instance by making sure that the largest value is positive.
    """
    # run CanICA many times (this is known to produce different results)
    canica = CanICA(
        n_components=4,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    for _ in range(3):
        canica.fit(canica_data)

        check_decomposition_estimator(canica, data_type)

        for mp in iter_img(canica.components_img_):
            mp = get_data(mp) if data_type == "nifti" else get_surface_data(mp)

            assert -mp.min() <= mp.max()
