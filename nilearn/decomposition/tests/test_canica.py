"""Test CanICA."""

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
<<<<<<< enh/surf_dict_learning
=======


def test_masker_attributes_with_fit(canica_data, mask_img):
    # Test base module at sub-class

    # Passing mask_img
    canica = CanICA(n_components=3, mask=mask_img, random_state=0)
    canica.fit(canica_data)

    assert canica.mask_img_ == canica.masker_.mask_img_

    # Passing masker
    masker = MultiNiftiMasker(mask_img=mask_img)
    canica = CanICA(n_components=3, mask=masker, random_state=0)
    canica.fit(canica_data)

    assert canica.mask_img_ == canica.masker_.mask_img_


def test_masker_attributes_passing_masker_arguments_to_estimator(
    affine_eye, canica_data
):
    canica = CanICA(
        n_components=3,
        target_affine=affine_eye,
        target_shape=(6, 8, 10),
        mask_strategy="background",
    )
    canica.fit(canica_data)


def test_components_img(canica_data, mask_img):
    n_components = 3

    canica = CanICA(n_components=n_components, mask=mask_img)
    canica.fit(canica_data)
    components_img = canica.components_img_

    assert isinstance(components_img, Nifti1Image)

    check_shape = canica_data[0].shape[:3] + (n_components,)

    assert components_img.shape, check_shape


def test_with_globbing_patterns_with_single_subject(mask_img, tmp_path):
    # single subject
    data, *_ = _make_canica_test_data(n_subjects=1)
    n_components = 3

    canica = CanICA(n_components=n_components, mask=mask_img)

    img = write_imgs_to_path(
        data[0], file_path=tmp_path, create_files=True, use_wildcards=True
    )
    canica.fit(img)
    components_img = canica.components_img_

    assert isinstance(components_img, Nifti1Image)

    # n_components = 3
    check_shape = data[0].shape[:3] + (3,)

    assert components_img.shape, check_shape


def test_with_globbing_patterns_with_single_subject_path(mask_img, tmp_path):
    # single subject but as a Path object
    data, *_ = _make_canica_test_data(n_subjects=1)
    n_components = 3

    canica = CanICA(n_components=n_components, mask=mask_img)

    tmp_file = tmp_path / "tmp.nii.gz"
    data[0].to_filename(tmp_file)

    canica.fit(tmp_file)


def test_with_globbing_patterns_with_multi_subjects(
    canica_data, mask_img, tmp_path
):
    # Multi subjects
    n_components = 3
    canica = CanICA(n_components=n_components, mask=mask_img)

    img = write_imgs_to_path(
        *canica_data, file_path=tmp_path, create_files=True, use_wildcards=True
    )
    canica.fit(img)
    components_img = canica.components_img_

    assert isinstance(components_img, Nifti1Image)

    # n_components = 3
    check_shape = canica_data[0].shape[:3] + (3,)

    assert components_img.shape, check_shape


def test_canica_score(canica_data, mask_img):
    # Multi subjects
    n_components = 10

    canica = CanICA(n_components=n_components, mask=mask_img, random_state=0)
    canica.fit(canica_data)

    # One score for all components
    scores = canica.score(canica_data, per_component=False)

    assert scores <= 1
    assert scores >= 0

    # Per component score
    scores = canica.score(canica_data, per_component=True)

    assert scores.shape, (n_components,)
    assert np.all(scores <= 1)
    assert np.all(scores >= 0)


def test_nifti_maps_masker_(canica_data, mask_img):
    """Check deprecation of nifti_maps_masker_."""
    n_components = 10

    canica = CanICA(n_components=n_components, mask=mask_img, random_state=0)
    canica.fit(canica_data)

    with pytest.deprecated_call(
        match="The 'nifti_maps_masker_' attribute is deprecated"
    ):
        canica.nifti_maps_masker_  # noqa: B018
>>>>>>> main
