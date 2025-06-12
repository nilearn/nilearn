"""Test CanICA."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.decomposition.canica import CanICA
from nilearn.image import get_data, iter_img
from nilearn.surface import SurfaceImage

ESTIMATORS_TO_CHECK = [CanICA()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_threshold_bound_error(canica_data_single_img):
    """Test that an error is raised when the threshold is higher \
    than the number of components.
    """
    with pytest.raises(ValueError, match="Threshold must not be higher"):
        canica = CanICA(n_components=4, threshold=5.0)
        canica.fit(canica_data_single_img)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_transform_and_fit_errors(decomposition_mask_img):
    """Test some errors of CanICA."""
    canica = CanICA(mask=decomposition_mask_img, n_components=3)

    # error when empty list of provided.
    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like or SurfaceImage objects as input, "
        "an empty list was given.",
    ):
        canica.fit([])

    # error is raised when no data is passed.
    with pytest.raises(
        TypeError,
        match="missing 1 required positional argument: 'imgs'",
    ):
        canica.fit()


@pytest.mark.parametrize("data_type", ["nifti"])
def test_percentile_range(rng, canica_data_single_img):
    """Test that a warning is given when thresholds are stressed."""
    edge_case = rng.integers(low=1, high=10)

    # stress thresholding via edge case
    canica = CanICA(n_components=edge_case, threshold=float(edge_case))

    with pytest.warns(UserWarning, match="obtained a critical threshold"):
        canica.fit(canica_data_single_img)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_canica_square_img(
    decomposition_mask_img, canica_components, canica_data
):
    """Check ???."""
    # We do a large number of inits to be sure to find the good match
    canica = CanICA(
        n_components=4,
        random_state=42,
        mask=decomposition_mask_img,
        smoothing_fwhm=0.0,
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


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_canica_single_subject_smoke(data_type, canica_data_single_img):
    """Check that canica runs on a single-subject dataset."""
    canica = CanICA(
        n_components=4, random_state=42, smoothing_fwhm=0.0, n_init=1
    )
    canica.fit(canica_data_single_img)
    if data_type == "nifti":
        assert isinstance(canica.mask_img_, Nifti1Image)
    elif data_type == "surface":
        assert isinstance(canica.mask_img_, SurfaceImage)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_component_sign(decomposition_mask_img, canica_data):
    """Check sign of extracted components.

    Regression test:
    We should have a heuristic that flips the sign of components in
    DictLearning to have more positive values than negative values, for
    instance by making sure that the largest value is positive.
    """
    # run CanICA many times (this is known to produce different results)
    canica = CanICA(
        n_components=4, random_state=42, mask=decomposition_mask_img
    )

    for _ in range(3):
        canica.fit(canica_data)
        for mp in iter_img(canica.components_img_):
            mp = get_data(mp)

            assert -mp.min() <= mp.max()


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_masker_attributes_with_fit(
    data_type,
    canica_data_single_img,
    decomposition_mask_img,
    decomposition_masker,
):
    """Test mask_img_ properly set when passing mask_img or masker."""
    # Passing mask_img
    canica = CanICA(
        n_components=3, mask=decomposition_mask_img, random_state=0
    )
    canica.fit(canica_data_single_img)

    if data_type == "nifti":
        assert isinstance(canica.mask_img_, Nifti1Image)
    else:
        assert isinstance(canica.mask_img_, SurfaceImage)

    assert canica.mask_img_ == canica.masker_.mask_img_

    # Passing masker
    canica = CanICA(n_components=3, mask=decomposition_masker, random_state=0)
    canica.fit(canica_data_single_img)

    assert canica.mask_img_ == canica.masker_.mask_img_


@pytest.mark.parametrize("data_type", ["nifti"])
def test_masker_attributes_passing_masker_arguments_to_estimator(
    affine_eye, canica_data_single_img
):
    """Smoke test that arguments for masker are passed along properly."""
    canica = CanICA(
        n_components=3,
        target_affine=affine_eye,
        target_shape=(6, 8, 10),
        mask_strategy="background",
    )
    canica.fit(canica_data_single_img)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_components_img(
    data_type, canica_data_single_img, decomposition_mask_img
):
    """Check content components_img_ after fitting."""
    n_components = 3

    canica = CanICA(n_components=n_components, mask=decomposition_mask_img)
    canica.fit(canica_data_single_img)
    components_img = canica.components_img_

    if data_type == "nifti":
        assert isinstance(components_img, Nifti1Image)
    else:
        assert isinstance(components_img, SurfaceImage)

    check_shape = canica_data_single_img.shape[:3] + (n_components,)

    assert components_img.shape, check_shape


@pytest.mark.parametrize("data_type", ["nifti"])
def test_with_globbing_patterns_with_single_subject(
    decomposition_mask_img, canica_data_single_img, tmp_path
):
    """Test CanICA with data on disk from a single subject with globbing."""
    n_components = 3

    canica = CanICA(n_components=n_components, mask=decomposition_mask_img)

    img = write_imgs_to_path(
        canica_data_single_img,
        file_path=tmp_path,
        create_files=True,
        use_wildcards=True,
    )
    canica.fit(img)
    components_img = canica.components_img_

    assert isinstance(components_img, Nifti1Image)

    # n_components = 3
    check_shape = canica_data_single_img.shape[:3] + (3,)

    assert components_img.shape, check_shape


@pytest.mark.parametrize("data_type", ["nifti"])
def test_with_globbing_patterns_with_single_subject_path(
    decomposition_mask_img, canica_data_single_img, tmp_path
):
    """Test CanICA with data on disk from a single subject as path."""
    n_components = 3

    canica = CanICA(n_components=n_components, mask=decomposition_mask_img)

    tmp_file = tmp_path / "tmp.nii.gz"
    canica_data_single_img.to_filename(tmp_file)

    canica.fit(tmp_file)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_with_globbing_patterns_with_multi_subjects(
    data_type, canica_data, decomposition_mask_img, tmp_path
):
    """Test CanICA with data on disk from multiple subject."""
    n_components = 3
    canica = CanICA(n_components=n_components, mask=decomposition_mask_img)

    img = write_imgs_to_path(
        *canica_data,
        file_path=tmp_path,
        create_files=True,
        use_wildcards=True,
    )
    canica.fit(img)
    components_img = canica.components_img_

    n_components = 3
    if data_type == "nifti":
        assert isinstance(components_img, Nifti1Image)
        check_shape = (*canica_data[0].shape[:3], n_components)
    else:
        assert isinstance(components_img, SurfaceImage)
        check_shape = (*canica_data[0].shape[0], n_components)

    assert components_img.shape, check_shape


@pytest.mark.parametrize("data_type", ["nifti"])
def test_canica_score(canica_data_single_img, decomposition_mask_img):
    """Check score of canonical ica."""
    # Multi subjects
    n_components = 10

    canica = CanICA(
        n_components=n_components, mask=decomposition_mask_img, random_state=0
    )
    canica.fit(canica_data_single_img)

    # One score for all components
    scores = canica.score(canica_data_single_img, per_component=False)

    assert scores <= 1
    assert scores >= 0

    # Per component score
    scores = canica.score(canica_data_single_img, per_component=True)

    assert scores.shape, (n_components,)
    assert np.all(scores <= 1)
    assert np.all(scores >= 0)


@pytest.mark.parametrize("data_type", ["nifti"])
def test_nifti_maps_masker_(canica_data_single_img, decomposition_mask_img):
    """Check depreacation of nifti_maps_masker_."""
    n_components = 10

    canica = CanICA(
        n_components=n_components, mask=decomposition_mask_img, random_state=0
    )
    canica.fit(canica_data_single_img)

    with pytest.deprecated_call(
        match="The 'nifti_maps_masker_' attribute is deprecated"
    ):
        canica.nifti_maps_masker_  # noqa: B018
