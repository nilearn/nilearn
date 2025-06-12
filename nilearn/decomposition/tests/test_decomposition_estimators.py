"""Common test for multi_pca, dict_learning, canica."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.decomposition.canica import CanICA
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.tests.conftest import (
    RANDOM_STATE,
    check_decomposition_estimator,
)

ESTIMATORS_TO_CHECK = [_MultiPCA(), DictLearning(), CanICA()]

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


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_fit_errors(
    data_type, decomposition_mask_img, decomposition_data, estimator
):
    """Fit and transform fail without the proper arguments."""
    est = estimator(mask=decomposition_mask_img, random_state=RANDOM_STATE)

    # Test if raises an error when empty list of provided.
    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like or SurfaceImage objects as input, "
        "an empty list was given.",
    ):
        est.fit([])

    # No mask provided
    est = estimator()
    # the default mask computation strategy 'epi' will result in an empty mask
    if data_type == "nifti":
        with pytest.raises(
            ValueError, match="The mask is invalid as it is empty"
        ):
            est.fit(decomposition_data)
    # but with surface images, the mask encompasses all vertices
    # so it should have the same number of True vertices as the vertices
    # in input images
    elif data_type == "surface":
        est.fit(decomposition_data)
        assert est.masker_.n_elements_ == decomposition_data[0].mesh.n_vertices


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_masker_attributes_with_fit(
    data_type,
    canica_data_single_img,
    decomposition_mask_img,
    decomposition_masker,
    estimator,
):
    """Test mask_img_ properly set when passing mask_img or masker."""
    # Passing mask_img
    est = estimator(
        n_components=3, mask=decomposition_mask_img, random_state=RANDOM_STATE
    )
    est.fit(canica_data_single_img)

    check_decomposition_estimator(est, data_type)

    # Passing masker
    canica = estimator(
        n_components=3, mask=decomposition_masker, random_state=RANDOM_STATE
    )
    canica.fit(canica_data_single_img)

    check_decomposition_estimator(canica, data_type)


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_pass_masker_arg_to_estimator(
    data_type, affine_eye, decomposition_data_single_img, estimator
):
    """Masker arguments are passed to the estimator without fail."""
    shape = (
        decomposition_data_single_img.shape[:3]
        if data_type == "nifti"
        else (decomposition_data_single_img.mesh.n_vertices,)
    )
    est = estimator(
        target_affine=affine_eye,
        target_shape=shape,
        n_components=3,
        mask_strategy="background",
        random_state=RANDOM_STATE,
    )

    # for surface we should get a warning about target_affine, target_shape
    # and mask_strategy being ignored
    if data_type == "surface":
        with pytest.warns(
            UserWarning, match="The following parameters are not relevant"
        ):
            est.fit(decomposition_data_single_img)
    elif data_type == "nifti":
        est.fit(decomposition_data_single_img)

    check_decomposition_estimator(est, data_type)


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti"])
def test_nifti_maps_masker_(canica_data_single_img, estimator):
    """Check depreacation of nifti_maps_masker_."""
    est = estimator()

    est.fit(canica_data_single_img)

    with pytest.deprecated_call(
        match="The 'nifti_maps_masker_' attribute is deprecated"
    ):
        est.nifti_maps_masker_  # noqa: B018


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_with_confounds(
    data_type, decomposition_mask_img, decomposition_data, estimator
):
    """Test of estimator with confounds.

    Output should be different with and without confounds.
    """
    confounds = [np.arange(10).reshape(5, 2)] * 8

    est = estimator(
        mask=decomposition_mask_img, n_components=3, random_state=RANDOM_STATE
    )

    est.fit(decomposition_data)

    check_decomposition_estimator(est, data_type)

    components = est.components_

    est.fit(decomposition_data, confounds=confounds)

    components_clean = est.components_

    assert_raises(
        AssertionError, assert_array_equal, components, components_clean
    )


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_single_subject_score(
    canica_data_single_img, decomposition_mask_img, data_type, estimator
):
    """Check content of scores after fitting."""
    n_components = 10
    est = estimator(
        n_components=n_components,
        mask=decomposition_mask_img,
        random_state=RANDOM_STATE,
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


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti"])
def test_single_subject_file(
    data_type, canica_data_single_img, estimator, tmp_path
):
    """Test with a single-subject dataset with globbing and path.

    Only for nifti as we cannot read surface from file.
    """
    est = estimator(
        n_components=4, random_state=RANDOM_STATE, smoothing_fwhm=0.0
    )
    # globbing
    img = write_imgs_to_path(
        canica_data_single_img,
        file_path=tmp_path,
        create_files=True,
        use_wildcards=True,
    )
    est.fit(img)

    check_decomposition_estimator(est, data_type)

    # path
    tmp_file = tmp_path / "tmp.nii.gz"
    canica_data_single_img.to_filename(tmp_file)

    est.fit(tmp_file)

    check_decomposition_estimator(est, data_type)


@pytest.mark.parametrize("estimator", [CanICA, _MultiPCA, DictLearning])
@pytest.mark.parametrize("data_type", ["nifti"])
@pytest.mark.parametrize("n_subjects", [1, 3])
def test_with_globbing_patterns(
    decomposition_mask_img,
    tmp_path,
    canica_data,
    data_type,
    estimator,
    n_subjects,  # noqa: ARG001
):
    """Check DictLearning can work with files on disk.

    Only for nifti as we cannot read surface from file.
    """
    est = estimator(n_components=3, mask=decomposition_mask_img)

    est.fit(canica_data)

    img = write_imgs_to_path(
        *canica_data, file_path=tmp_path, create_files=True, use_wildcards=True
    )

    est.fit(img)

    check_decomposition_estimator(est, data_type)
