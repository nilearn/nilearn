"""Test the multi-PCA module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.surface import SurfaceImage

ESTIMATORS_TO_CHECK = [_MultiPCA()]

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


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_check_masker_attributes(
    data_type,
    decomposition_mask_img,
    decomposition_data,
):
    """Check fit creates proper mask_img_ attributes."""
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img, n_components=3, random_state=0
    )
    multi_pca.fit(decomposition_data)

    if data_type == "nifti":
        assert isinstance(multi_pca.mask_img_, Nifti1Image)
    elif data_type == "surface":
        assert isinstance(multi_pca.mask_img_, SurfaceImage)

    assert multi_pca.mask_img_ == multi_pca.masker_.mask_img_


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
@pytest.mark.parametrize("length", [1, 2])
def test_multi_pca(
    length,
    data_type,
    decomposition_mask_img,
    decomposition_data,
):
    """Components are the same if we put twice the same data, \
       and that fit output is deterministic.
    """
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img, n_components=3, random_state=0
    )
    multi_pca.fit(decomposition_data)

    if data_type == "nifti":
        assert isinstance(multi_pca.mask_img_, Nifti1Image)
    elif data_type == "surface":
        assert isinstance(multi_pca.mask_img_, SurfaceImage)

    components1 = multi_pca.components_
    components2 = multi_pca.fit(length * decomposition_data).components_

    if length == 1:
        assert_array_equal(components1, components2)
    else:
        assert_array_almost_equal(components1, components2)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_with_confounds(
    data_type,
    decomposition_mask_img,
    decomposition_data,
):
    """Test of _MultiPCA with confounds."""
    confounds = [np.arange(10).reshape(5, 2)] * 8

    multi_pca = _MultiPCA(
        mask=decomposition_mask_img, n_components=3, random_state=0
    )
    multi_pca.fit(decomposition_data)

    if data_type == "nifti":
        assert isinstance(multi_pca.mask_img_, Nifti1Image)
    elif data_type == "surface":
        assert isinstance(multi_pca.mask_img_, SurfaceImage)

    components = multi_pca.components_

    multi_pca.fit(decomposition_data, confounds=confounds)

    components_clean = multi_pca.components_

    assert_raises(
        AssertionError, assert_array_equal, components, components_clean
    )


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_errors(
    data_type, decomposition_mask_img, decomposition_data
):
    """Fit and transform fail without the proper arguments."""
    multi_pca = _MultiPCA(mask=decomposition_mask_img)

    # Smoke test to fit with no img
    with pytest.raises(TypeError, match="missing 1 required positional"):
        multi_pca.fit()

    # Test if raises an error when empty list of provided.
    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like objects as input, "
        "an empty list was given.",
    ):
        multi_pca.fit([])

    # No mask provided
    multi_pca = _MultiPCA()
    # the default mask computation strategy 'epi' will result in an empty mask
    if data_type == "nifti":
        with pytest.raises(
            ValueError, match="The mask is invalid as it is empty"
        ):
            multi_pca.fit(decomposition_data)
    # but with surface images, the mask encompasses all vertices
    # so it should have the same number of True vertices as the vertices
    # in input images
    elif data_type == "surface":
        multi_pca.fit(decomposition_data)
        assert (
            multi_pca.masker_.n_elements_
            == decomposition_data[0].mesh.n_vertices
        )


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_with_masker(
    data_type,
    decomposition_masker,
    decomposition_data,
):
    """Multi-pca can run with a masker."""
    multi_pca = _MultiPCA(mask=decomposition_masker, n_components=3)
    multi_pca.fit(decomposition_data)

    if data_type == "nifti":
        assert isinstance(multi_pca.masker_, NiftiMasker)
    elif data_type == "surface":
        assert isinstance(multi_pca.masker_, SurfaceMasker)

    assert multi_pca.mask_img_ == multi_pca.masker_.mask_img_


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_with_masker_without_cca_smoke(
    data_type,
    decomposition_masker,
    decomposition_data,
):
    """Multi-pca can run with a masker \
        and without canonical correlation analysis.
    """
    multi_pca = _MultiPCA(
        mask=decomposition_masker,
        do_cca=False,
        n_components=3,
    )
    multi_pca.fit(decomposition_data[:2])

    if data_type == "nifti":
        assert isinstance(multi_pca.masker_, NiftiMasker)
    elif data_type == "surface":
        assert isinstance(multi_pca.masker_, SurfaceMasker)

    # Smoke test the transform and inverse_transform
    multi_pca.inverse_transform(multi_pca.transform(decomposition_data[-2:]))


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_pass_masker_arg_to_estimator_smoke(
    data_type, affine_eye, decomposition_data_single_img
):
    """Masker arguments are passed to the estimator without fail."""
    shape = (
        decomposition_data_single_img.shape[:3]
        if data_type == "nifti"
        else (decomposition_data_single_img.mesh.n_vertices,)
    )
    multi_pca = _MultiPCA(
        target_affine=affine_eye,
        target_shape=shape,
        n_components=3,
        mask_strategy="background",
    )

    # for surface we should get a warning about target_affine, target_shape
    # and mask_strategy being ignored
    if data_type == "surface":
        with pytest.warns(
            UserWarning, match="The following parameters are not relevant"
        ):
            multi_pca.fit(decomposition_data_single_img)
    elif data_type == "nifti":
        multi_pca.fit(decomposition_data_single_img)


@pytest.mark.parametrize("with_activation", [False])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_score_gt_0_lt_1(
    decomposition_mask_img,
    decomposition_data,
    data_type,  # noqa: ARG001
    with_activation,  # noqa: ARG001
):
    """Test that MultiPCA score is between zero and one."""
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img,
        random_state=0,
        memory_level=0,
        n_components=3,
    )
    multi_pca.fit(decomposition_data)
    s = multi_pca.score(decomposition_data)

    assert np.all(s <= 1)
    assert np.all(s >= 0)


@pytest.mark.parametrize("with_activation", [False])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_score_single_subject(
    decomposition_mask_img,
    decomposition_data_single_img,
    data_type,  # noqa: ARG001
    with_activation,  # noqa: ARG001
):
    """Multi-pca can run on single subject data."""
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img,
        random_state=0,
        memory_level=0,
        n_components=3,
    )
    multi_pca.fit(decomposition_data_single_img)
    s = multi_pca.score(decomposition_data_single_img)

    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("with_activation", [False])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_score_single_subject_n_components(
    data_type,
    decomposition_mask_img,
    decomposition_data_single_img,
    with_activation,  # noqa: ARG001
):
    """Score is one for n_components == n_sample \
       in single subject configuration.
    """
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img,
        random_state=0,
        memory_level=0,
        n_components=5,
    )
    multi_pca.fit(decomposition_data_single_img)
    s = multi_pca.score(decomposition_data_single_img)

    assert_almost_equal(s, 1.0, 1)

    # Per component score
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img,
        random_state=0,
        memory_level=0,
        n_components=5,
    )
    multi_pca.fit(decomposition_data_single_img)
    if data_type == "nifti":
        masker = NiftiMasker(decomposition_mask_img).fit()
    elif data_type == "surface":
        masker = SurfaceMasker(decomposition_mask_img).fit()
    s = multi_pca._raw_score(
        masker.transform(decomposition_data_single_img), per_component=True
    )

    assert s.shape == (5,)
    assert np.all(s <= 1)
    assert np.all(s >= 0)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_components_img(data_type, decomposition_mask_img, decomposition_data):
    """Check content of components_img_ after fit."""
    n_components = 3

    multi_pca = _MultiPCA(
        mask=decomposition_mask_img, n_components=n_components, random_state=0
    )
    multi_pca.fit(decomposition_data)
    components_img = multi_pca.components_img_

    if data_type == "nifti":
        assert isinstance(components_img, Nifti1Image)
        check_shape = (*multi_pca.masker_.mask_img_.shape, n_components)
        assert len(components_img.shape) == 4
    elif data_type == "surface":
        assert isinstance(components_img, SurfaceImage)
        check_shape = (multi_pca.masker_.mask_img_.shape[-1], n_components)
        assert len(components_img.shape) == 2

    assert components_img.shape == check_shape


@pytest.mark.parametrize(
    "imgs",
    [
        [generate_fake_fmri()[0]],
        [generate_fake_fmri()[0], generate_fake_fmri()[0]],
    ],
)
def test_with_globbing_patterns_on_one_or_several_images(imgs, tmp_path):
    """Check that _MultiPCA can work with one or more 4D images from disk."""
    multi_pca = _MultiPCA(n_components=3)

    filenames = write_imgs_to_path(
        *imgs, file_path=tmp_path, create_files=True, use_wildcards=True
    )

    multi_pca.fit(filenames)

    components_img = multi_pca.components_img_
    assert isinstance(components_img, Nifti1Image)

    n_components = 3
    check_shape = (*imgs[0].shape[:3], n_components)
    assert components_img.shape == check_shape
    assert len(components_img.shape) == 4
