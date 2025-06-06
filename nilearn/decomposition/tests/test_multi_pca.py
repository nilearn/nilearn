"""Test the multi-PCA module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._utils.testing import write_imgs_to_path
from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.decomposition.tests.test_base import (
    make_data_to_reduce,
    make_masker,
)
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.surface import SurfaceImage


def img_4d():
    data_4d = np.zeros((40, 40, 40, 3))
    data_4d[20, 20, 20] = 1
    return Nifti1Image(data_4d, affine=np.eye(4))


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
def test_multi_pca_check_masker_attributes(data_type):
    imgs, mask_img = make_data_to_reduce(data_type=data_type)
    multi_pca = _MultiPCA(mask=mask_img, n_components=3, random_state=0)
    multi_pca.fit(imgs)

    assert multi_pca.mask_img_ == multi_pca.masker_.mask_img_


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
@pytest.mark.parametrize("length", [1, 2])
def test_multi_pca(length, data_type):
    """Components are the same if we put twice the same data, \
       and that fit output is deterministic.
    """
    imgs, mask_img = make_data_to_reduce(data_type=data_type)
    multi_pca = _MultiPCA(mask=mask_img, n_components=3, random_state=0)
    multi_pca.fit(imgs)

    components1 = multi_pca.components_
    components2 = multi_pca.fit(length * imgs).components_

    if length == 1:
        np.testing.assert_array_equal(components1, components2)
    else:
        np.testing.assert_array_almost_equal(components1, components2)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_with_confounds_smoke(data_type):
    imgs, mask_img = make_data_to_reduce(data_type=data_type)
    confounds = [np.arange(10).reshape(5, 2)] * 8

    multi_pca = _MultiPCA(mask=mask_img, n_components=3, random_state=0)
    multi_pca.fit(imgs, confounds=confounds)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_errors(data_type):
    """Fit and transform fail without the proper arguments."""
    imgs, mask_img = make_data_to_reduce(data_type=data_type)
    multi_pca = _MultiPCA(mask=mask_img)

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
            multi_pca.fit(imgs)
    # but with surface images, the mask encompasses all vertices
    # so it should have the same number of True vertices as the vertices
    # in input images
    elif data_type == "surface":
        multi_pca.fit(imgs)
        assert multi_pca.masker_.n_elements_ == imgs[0].mesh.n_vertices


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_with_masker(data_type):
    """Multi-pca can run with a masker."""
    imgs, _ = make_data_to_reduce(data_type=data_type)
    masker = make_masker(data_type=data_type)

    multi_pca = _MultiPCA(mask=masker, n_components=3)
    multi_pca.fit(imgs)

    assert multi_pca.mask_img_ == multi_pca.masker_.mask_img_


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_with_masker_without_cca_smoke(data_type):
    """Multi-pca can run with a masker \
        and without canonical correlation analysis.
    """
    masker = make_masker(data_type=data_type)
    data, _ = make_data_to_reduce(data_type=data_type)

    multi_pca = _MultiPCA(
        mask=masker,
        do_cca=False,
        n_components=3,
    )
    multi_pca.fit(data[:2])

    # Smoke test the transform and inverse_transform
    multi_pca.inverse_transform(multi_pca.transform(data[-2:]))


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_pass_masker_arg_to_estimator_smoke(data_type, affine_eye):
    """Masker arguments are passed to the estimator without fail."""
    data, _ = make_data_to_reduce(data_type=data_type)
    shape = (
        data[0].shape[:3]
        if data_type == "nifti"
        else (data[0].mesh.n_vertices,)
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
            multi_pca.fit(data)
    elif data_type == "nifti":
        multi_pca.fit(data)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_score_gt_0_lt_1(data_type):
    """Test that MultiPCA score is between zero and one."""
    data, mask_img = make_data_to_reduce(
        with_activation=False, data_type=data_type
    )

    multi_pca = _MultiPCA(
        mask=mask_img, random_state=0, memory_level=0, n_components=3
    )
    multi_pca.fit(data)
    s = multi_pca.score(data)

    assert np.all(s <= 1)
    assert np.all(s >= 0)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_score_single_subject(data_type):
    """Multi-pca can run on single subject data."""
    data, mask_img = make_data_to_reduce(
        with_activation=False, data_type=data_type
    )

    multi_pca = _MultiPCA(
        mask=mask_img, random_state=0, memory_level=0, n_components=3
    )
    multi_pca.fit(data[0])
    s = multi_pca.score(data[0])

    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_score_single_subject_n_components(data_type):
    """Score is one for n_components == n_sample \
       in single subject configuration.
    """
    data, mask_img = make_data_to_reduce(
        with_activation=False, data_type=data_type
    )
    multi_pca = _MultiPCA(
        mask=mask_img, random_state=0, memory_level=0, n_components=5
    )
    multi_pca.fit(data[0])
    s = multi_pca.score(data[0])

    assert_almost_equal(s, 1.0, 1)

    # Per component score
    multi_pca = _MultiPCA(
        mask=mask_img, random_state=0, memory_level=0, n_components=5
    )
    multi_pca.fit(data[0])
    if data_type == "nifti":
        masker = NiftiMasker(mask_img).fit()
    elif data_type == "surface":
        masker = SurfaceMasker(mask_img).fit()
    s = multi_pca._raw_score(masker.transform(data[0]), per_component=True)

    assert s.shape == (5,)
    assert np.all(s <= 1)
    assert np.all(s >= 0)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_components_img(data_type):
    n_components = 3
    data, mask_img = make_data_to_reduce(data_type=data_type)

    multi_pca = _MultiPCA(
        mask=mask_img, n_components=n_components, random_state=0
    )
    multi_pca.fit(data)
    components_img = multi_pca.components_img_

    if data_type == "nifti":
        assert isinstance(components_img, Nifti1Image)
        check_shape = data[0].shape[:3] + (n_components,)
        assert components_img.shape == check_shape
        assert len(components_img.shape) == 4
    elif data_type == "surface":
        assert isinstance(components_img, SurfaceImage)
        check_shape = (data[0].mesh.n_vertices, n_components)
        assert components_img.shape == check_shape
        assert len(components_img.shape) == 2


@pytest.mark.parametrize("imgs", [[img_4d()], [img_4d(), img_4d()]])
def test_with_globbing_patterns_on_one_or_several_images(imgs, tmp_path):
    multi_pca = _MultiPCA(n_components=3)

    filenames = write_imgs_to_path(
        *imgs, file_path=tmp_path, create_files=True, use_wildcards=True
    )

    multi_pca.fit(filenames)

    components_img = multi_pca.components_img_
    assert isinstance(components_img, Nifti1Image)

    # n_components = 3
    check_shape = img_4d().shape[:3] + (3,)
    assert components_img.shape == check_shape
    assert len(components_img.shape) == 4
