"""Common test for multi_pca, dict_learning, canica."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises
from sklearn import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.decomposition import CanICA, DictLearning
from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.decomposition.tests.conftest import (
    N_SAMPLES,
    N_SUBJECTS,
    RANDOM_STATE,
    check_decomposition_estimator,
)

ESTIMATORS_TO_CHECK = [
    DictLearning(verbose=0),
    CanICA(verbose=0),
]

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


@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_fit_errors(
    data_type, decomposition_images, estimator, decomposition_mask_img
):
    """Fit fail without the proper arguments."""
    est = estimator(
        smoothing_fwhm=None,
    )

    # Test if raises an error when empty list of provided.
    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like or SurfaceImage objects as input, "
        "an empty list was given.",
    ):
        est.fit([])

    # No mask provided
    est = estimator(
        smoothing_fwhm=None,
    )
    # the default mask computation strategy 'epi' will result in an empty mask
    if data_type == "nifti":
        with pytest.raises(
            ValueError, match="The mask is invalid as it is empty"
        ):
            est.fit(decomposition_images)
    # but with surface images, the mask encompasses all vertices
    # so it should have the same number of True vertices as the vertices
    # in input images
    elif data_type == "surface":
        est.fit(decomposition_images)
        assert (
            est.masker_.n_elements_ == decomposition_images[0].mesh.n_vertices
        )

    # mismatch len confounds and input to fit
    est = estimator(
        n_components=3,
        mask=decomposition_mask_img,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    confounds = (
        [np.arange(N_SAMPLES * 2).reshape(N_SAMPLES, 2)]
        * len(decomposition_images)
        * 2
    )
    with pytest.raises(
        ValueError,
        match="Number of confounds .* must match number of images .*",
    ):
        est.fit(decomposition_images, confounds=confounds)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_masker_attributes_with_fit(
    data_type,
    canica_data,
    decomposition_mask_img,
    decomposition_masker,
    estimator,
):
    """Test mask_img_ properly set when passing mask_img or masker."""
    # Passing mask_img
    est = estimator(
        n_components=3,
        mask=decomposition_mask_img,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )
    est.fit(canica_data)

    check_decomposition_estimator(est, data_type)

    # Passing masker
    est = estimator(
        n_components=3,
        mask=decomposition_masker,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    with pytest.warns(UserWarning, match="overriding estimator parameter"):
        est.fit(canica_data)

    check_decomposition_estimator(est, data_type)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_transform(
    data_type,  # noqa: ARG001
    canica_data,
    estimator,
):
    """Test transform and inverse transform."""
    est = estimator(
        n_components=3,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    est.fit(canica_data)

    signals = est.transform(canica_data)

    assert isinstance(signals, list)
    for x in signals:
        assert isinstance(x, np.ndarray)

    # output of fit + transform == output fit transform
    est = clone(est)
    signals_2 = est.fit_transform(canica_data)

    assert_array_equal(signals, signals_2)

    # smoke test
    est.inverse_transform(signals)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_transform_confounds(
    data_type,
    canica_data,
    estimator,
):
    """Test transform with confounds give different results."""
    est = estimator(
        n_components=3,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )
    if data_type == "surface" and isinstance(est, DictLearning):
        pytest.skip(
            "dummy data for surface give empty signals with DictLearning"
        )

    est.fit(canica_data)

    signals = est.transform(canica_data)

    if data_type == "surface":
        n_samples = canica_data[0].shape[1]
    else:
        n_samples = canica_data[0].shape[3]

    confounds = [np.arange(n_samples * 2).reshape(n_samples, 2)] * len(
        canica_data
    )

    signals_confounds = est.transform(canica_data, confounds=confounds)

    assert_raises(
        AssertionError, assert_array_equal, signals, signals_confounds
    )


@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_transform_single_image(
    data_type,  # noqa: ARG001
    canica_data_single_img,
    estimator,
):
    """Test transform on single image.

    Passing a single image instead of list to transform should work
    but still return a list.
    """
    est = estimator(
        n_components=3,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    assert not isinstance(canica_data_single_img, list)

    est.fit(canica_data_single_img)

    signals = est.transform(canica_data_single_img)

    assert isinstance(signals, list)


@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_transform_errors(
    data_type,  # noqa: ARG001
    estimator,
    canica_data,
):
    """Test errors transform and inverse transform."""
    est = estimator(
        n_components=3,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    est.fit(canica_data)

    confounds = (
        [np.arange(N_SAMPLES * 2).reshape(N_SAMPLES, 2)] * len(canica_data) * 2
    )
    with pytest.raises(
        ValueError,
        match="Number of confounds .* must match number of images .*",
    ):
        est.transform(canica_data, confounds=confounds)


@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_pass_masker_arg_to_estimator(
    data_type, affine_eye, canica_data, estimator
):
    """Masker arguments are passed to the estimator without fail."""
    shape = (
        canica_data[0].shape[:3]
        if data_type == "nifti"
        else (canica_data[0].mesh.n_vertices,)
    )
    est = estimator(
        target_affine=affine_eye,
        target_shape=shape,
        n_components=3,
        mask_strategy="background",
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    # for surface we should get a warning about target_affine, target_shape
    # and mask_strategy being ignored
    if data_type == "surface":
        with pytest.warns(
            UserWarning, match="The following parameters are not relevant"
        ):
            est.fit(canica_data)
    elif data_type == "nifti":
        est.fit(canica_data)

    check_decomposition_estimator(est, data_type)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_nifti_maps_masker_(
    data_type,  # noqa: ARG001
    canica_data,
    estimator,
):
    """Check deprecation of nifti_maps_masker_."""
    est = estimator(
        n_components=3,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    est.fit(canica_data)

    with pytest.deprecated_call(
        match="The 'nifti_maps_masker_' attribute is deprecated"
    ):
        est.nifti_maps_masker_  # noqa: B018


# TODO passing confounds does not affect output with CanICA, DictLearning
# @pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("estimator", [_MultiPCA])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_with_confounds(
    data_type, decomposition_images, decomposition_mask_img, estimator
):
    """Test of estimator with confounds.

    Output should be different with and without confounds.
    """
    confounds = [np.arange(N_SAMPLES * 2).reshape(N_SAMPLES, 2)] * N_SUBJECTS

    est = estimator(
        n_components=3,
        random_state=RANDOM_STATE,
        mask=decomposition_mask_img,
        smoothing_fwhm=None,
    )

    est.fit(decomposition_images)

    check_decomposition_estimator(est, data_type)

    components = est.components_

    est = estimator(
        n_components=3, random_state=RANDOM_STATE, mask=decomposition_mask_img
    )
    est.fit(decomposition_images, confounds=confounds)

    components_clean = est.components_

    assert_raises(
        AssertionError, assert_array_equal, components, components_clean
    )


@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_single_subject_score(canica_data_single_img, data_type, estimator):
    """Check content of scores after fitting."""
    n_components = 3

    # quick sanity check to avoid some tests failures if
    # n_components < N_SAMPLES
    assert n_components < N_SAMPLES

    est = estimator(
        n_components=n_components,
        random_state=RANDOM_STATE,
        smoothing_fwhm=None,
    )

    est.fit(canica_data_single_img)

    check_decomposition_estimator(est, data_type)

    # One score for all components
    scores = est.score(canica_data_single_img, per_component=False)

    assert isinstance(scores, float)
    assert 0 <= scores <= 1

    # Per component score
    scores = est.score(canica_data_single_img, per_component=True)

    assert scores.shape, (n_components,)
    assert np.all(scores <= 1)
    assert np.all(scores >= 0)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti"])
def test_single_subject_file(
    data_type, canica_data_single_img, estimator, tmp_path
):
    """Test with a single-subject dataset with globbing and path.

    Only for nifti as we cannot read surface from file.
    """
    # globbing
    est = estimator(n_components=4, random_state=RANDOM_STATE)
    img = write_imgs_to_path(
        canica_data_single_img,
        file_path=tmp_path,
        create_files=True,
        use_wildcards=True,
    )
    est.fit(img)

    check_decomposition_estimator(est, data_type)

    # smoke test transform
    est.transform(img)

    # path
    est = clone(est)
    tmp_file = tmp_path / "tmp.nii.gz"
    canica_data_single_img.to_filename(tmp_file)

    est.fit(tmp_file)

    check_decomposition_estimator(est, data_type)

    # smoke test transform
    est.transform(tmp_file)


@pytest.mark.timeout(0)
@pytest.mark.parametrize("estimator", [CanICA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti"])
@pytest.mark.parametrize("n_subjects", [1, 3])
def test_with_globbing_patterns(
    tmp_path,
    canica_data,
    data_type,
    estimator,
    n_subjects,  # noqa: ARG001
):
    """Check DictLearning can work with files on disk.

    Only for nifti as we cannot read surface from file.
    """
    est = estimator(n_components=3)

    est.fit(canica_data)

    img = write_imgs_to_path(
        *canica_data, file_path=tmp_path, create_files=True, use_wildcards=True
    )

    est.fit(img)

    check_decomposition_estimator(est, data_type)

    # smoke test transform and inverse transform
    est.transform(img)
