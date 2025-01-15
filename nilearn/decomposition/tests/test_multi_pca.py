"""Test the multi-PCA module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal

from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.testing import write_imgs_to_path
from nilearn.conftest import _affine_eye, _rng
from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.maskers import MultiNiftiMasker, NiftiMasker

SHAPE = (6, 8, 10)


def img_4d():
    data_4d = np.zeros((40, 40, 40, 3))
    data_4d[20, 20, 20] = 1
    return Nifti1Image(data_4d, affine=np.eye(4))


def _make_multi_pca_test_data(with_activation=True):
    """Create a multi-subject dataset with or without activation."""
    shape = (6, 8, 10, 5)
    affine = _affine_eye()
    rng = _rng()
    n_sub = 4

    data = []
    for _ in range(n_sub):
        this_data = rng.normal(size=shape)
        if with_activation:
            this_data[2:4, 2:4, 2:4, :] += 10
        data.append(Nifti1Image(this_data, affine))

    mask_img = Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)

    return data, mask_img, shape, affine


@pytest.fixture(scope="module")
def mask_img():
    return Nifti1Image(np.ones(SHAPE, dtype=np.int8), _affine_eye())


@pytest.fixture(scope="module")
def multi_pca_data():
    return _make_multi_pca_test_data()[0]


extra_valid_checks = [
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_estimators_unfitted",
    "check_get_params_invariance",
    "check_no_attributes_set_in_init",
    "check_transformers_unfitted",
    "check_transformer_n_iter",
    "check_parameters_default_constructible",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[_MultiPCA()], extra_valid_checks=extra_valid_checks
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[_MultiPCA()],
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_multi_pca_check_masker_attributes(multi_pca_data, mask_img):
    multi_pca = _MultiPCA(mask=mask_img, n_components=3, random_state=0)
    multi_pca.fit(multi_pca_data)

    assert multi_pca.mask_img_ == mask_img
    assert multi_pca.mask_img_ == multi_pca.masker_.mask_img_


def test_multi_pca(multi_pca_data, mask_img):
    """Components are the same if we put twice the same data, \
       and that fit output is deterministic.
    """
    multi_pca = _MultiPCA(mask=mask_img, n_components=3, random_state=0)
    multi_pca.fit(multi_pca_data)

    components1 = multi_pca.components_
    components2 = multi_pca.fit(multi_pca_data).components_
    components3 = multi_pca.fit(2 * multi_pca_data).components_

    np.testing.assert_array_equal(components1, components2)
    np.testing.assert_array_almost_equal(components1, components3)


def test_multi_pca_with_confounds_smoke(multi_pca_data, mask_img):
    confounds = [np.arange(10).reshape(5, 2)] * 8

    multi_pca = _MultiPCA(mask=mask_img, n_components=3, random_state=0)
    multi_pca.fit(multi_pca_data, confounds=confounds)


def test_multi_pca_componenent_errors(mask_img):
    """Test that a ValueError is raised \
    if the number of components is too low.
    """
    multi_pca = _MultiPCA(mask=mask_img)
    with pytest.raises(
        ValueError, match="Object has no components_ attribute."
    ):
        multi_pca._check_components_()


def test_multi_pca_errors(multi_pca_data, mask_img):
    """Fit and transform fail without the proper arguments."""
    multi_pca = _MultiPCA(mask=mask_img)

    # Smoke test to fit with no img
    with pytest.raises(TypeError, match="missing 1 required positional"):
        multi_pca.fit()

    # transform before fit raises an error
    with pytest.raises(
        ValueError,
        match="Object has no components_ attribute. This is "
        "probably because fit has not been called",
    ):
        multi_pca.transform(multi_pca_data)

    # Test if raises an error when empty list of provided.
    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like objects as input, "
        "an empty list was given.",
    ):
        multi_pca.fit([])

    # No mask provided
    multi_pca = _MultiPCA()
    with pytest.raises(ValueError, match="The mask is invalid as it is empty"):
        multi_pca.fit(multi_pca_data)


def test_multi_pca_with_masker(multi_pca_data):
    """Multi-pca can run with a masker."""
    masker = MultiNiftiMasker()

    multi_pca = _MultiPCA(mask=masker, n_components=3)
    multi_pca.fit(multi_pca_data)

    assert multi_pca.mask_img_ == multi_pca.masker_.mask_img_


def test_multi_pca_with_masker_without_cca_smoke(multi_pca_data):
    """Multi-pca can run with a masker \
        and without canonical correlation analysis.
    """
    masker = MultiNiftiMasker(mask_args={"opening": 0})

    multi_pca = _MultiPCA(
        mask=masker,
        do_cca=False,
        n_components=3,
    )
    multi_pca.fit(multi_pca_data[:2])

    # Smoke test the transform and inverse_transform
    multi_pca.inverse_transform(multi_pca.transform(multi_pca_data[-2:]))


def test_multi_pca_pass_masker_arg_to_estimator_smoke():
    """Masker arguments are passed to the estimator without fail."""
    data, _, shape, affine = _make_multi_pca_test_data()

    multi_pca = _MultiPCA(
        target_affine=affine,
        target_shape=shape[:3],
        n_components=3,
        mask_strategy="background",
    )
    multi_pca.fit(data)


def test_multi_pca_score_gt_0_lt_1(mask_img):
    """Test that MultiPCA score is between zero and one."""
    data, _, _, _ = _make_multi_pca_test_data(with_activation=False)

    multi_pca = _MultiPCA(
        mask=mask_img, random_state=0, memory_level=0, n_components=3
    )
    multi_pca.fit(data)
    s = multi_pca.score(data)

    assert np.all(s <= 1)
    assert np.all(s >= 0)


def test_multi_pca_score_single_subject(mask_img):
    """Multi-pca can run on single subject data."""
    data, _, _, _ = _make_multi_pca_test_data(with_activation=False)

    multi_pca = _MultiPCA(
        mask=mask_img, random_state=0, memory_level=0, n_components=3
    )
    multi_pca.fit(data[0])
    s = multi_pca.score(data[0])

    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_multi_pca_score_single_subject_n_components(mask_img):
    """Score is one for n_components == n_sample \
       in single subject configuration.
    """
    data, _, _, _ = _make_multi_pca_test_data(with_activation=False)
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
    masker = NiftiMasker(mask_img).fit()
    s = multi_pca._raw_score(masker.transform(data[0]), per_component=True)

    assert s.shape == (5,)
    assert np.all(s <= 1)
    assert np.all(s >= 0)


def test_components_img(multi_pca_data, mask_img):
    n_components = 3

    multi_pca = _MultiPCA(
        mask=mask_img, n_components=n_components, random_state=0
    )
    multi_pca.fit(multi_pca_data)
    components_img = multi_pca.components_img_

    assert isinstance(components_img, Nifti1Image)

    check_shape = multi_pca_data[0].shape[:3] + (n_components,)

    assert components_img.shape == check_shape
    assert len(components_img.shape) == 4


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
