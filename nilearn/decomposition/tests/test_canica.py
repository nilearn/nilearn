"""Test CanICA."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal

from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.testing import write_imgs_to_path
from nilearn.conftest import _affine_eye, _rng
from nilearn.decomposition.canica import CanICA
from nilearn.image import get_data, iter_img
from nilearn.maskers import MultiNiftiMasker

SHAPE = (30, 30, 5)

N_SUBJECTS = 2


def _make_data_from_components(
    components,
    affine=None,
    shape=SHAPE,
    rng=None,
    n_subjects=N_SUBJECTS,
):
    if affine is None:
        affine = _affine_eye()
    data = []
    if rng is None:
        rng = _rng()
    background = -0.01 * rng.normal(size=shape) - 2
    background = background[..., np.newaxis]
    for _ in range(n_subjects):
        this_data = np.dot(rng.normal(size=(40, 4)), components)
        this_data += 0.01 * rng.normal(size=this_data.shape)
        # Get back into 3D for CanICA
        this_data = np.reshape(this_data, (40, *shape))
        this_data = np.rollaxis(this_data, 0, 4)
        # Put the border of the image to zero, to mimic a brain image
        this_data[:5] = background[:5]
        this_data[-5:] = background[-5:]
        this_data[:, :5] = background[:, :5]
        this_data[:, -5:] = background[:, -5:]
        data.append(Nifti1Image(this_data, affine))
    return data


def _make_canica_components(shape):
    # Create two images with "activated regions"
    component1 = np.zeros(shape)
    component1[:5, :10] = 1
    component1[5:10, :10] = -1

    component2 = np.zeros(shape)
    component2[:5, -10:] = 1
    component2[5:10, -10:] = -1

    component3 = np.zeros(shape)
    component3[-5:, -10:] = 1
    component3[-10:-5, -10:] = -1

    component4 = np.zeros(shape)
    component4[-5:, :10] = 1
    component4[-10:-5, :10] = -1

    return np.vstack(
        (
            component1.ravel(),
            component2.ravel(),
            component3.ravel(),
            component4.ravel(),
        )
    )


def _make_canica_test_data(rng=None, n_subjects=N_SUBJECTS, noisy=True):
    if rng is None:
        # Use legacy generator for sklearn compatibility
        rng = np.random.RandomState(42)
    components = _make_canica_components(SHAPE)
    if noisy:  # Creating noisy non positive data
        components[rng.standard_normal(components.shape) > 0.8] *= -2.0

    for mp in components:
        assert mp.max() <= -mp.min()  # Goal met ?

    # Create a "multi-subject" dataset
    data = _make_data_from_components(
        components, _affine_eye(), SHAPE, rng=rng, n_subjects=n_subjects
    )

    return data, components, rng


@pytest.fixture(scope="module")
def mask_img():
    mask = np.ones(SHAPE)
    mask[:5] = 0
    mask[-5:] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0
    mask[..., -2:] = 0
    mask[..., :2] = 0
    return Nifti1Image(mask, _affine_eye())


@pytest.fixture(scope="module")
def canica_data():
    return _make_canica_test_data()[0]


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
        estimator=[CanICA()], extra_valid_checks=extra_valid_checks
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[CanICA()],
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_threshold_bound_error(canica_data):
    """Test that an error is raised when the threshold is higher \
    than the number of components.
    """
    with pytest.raises(ValueError, match="Threshold must not be higher"):
        canica = CanICA(n_components=4, threshold=5.0)
        canica.fit(canica_data)


def test_transform_and_fit_errors(canica_data, mask_img):
    canica = CanICA(mask=mask_img, n_components=3)

    with pytest.raises(
        ValueError,
        match="Object has no components_ attribute. "
        "This is probably because fit has not been called.",
    ):
        canica.transform(canica_data)

    # error when empty list of provided.
    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like objects as input, "
        "an empty list was given.",
    ):
        canica.fit([])

    # error is raised when no data is passed.
    with pytest.raises(
        TypeError,
        match="missing 1 required positional argument: 'imgs'",
    ):
        canica.fit()


def test_percentile_range(rng, canica_data):
    """Test that a warning is given when thresholds are stressed."""
    edge_case = rng.integers(low=1, high=10)

    # stress thresholding via edge case
    canica = CanICA(n_components=edge_case, threshold=float(edge_case))

    with pytest.warns(UserWarning, match="obtained a critical threshold"):
        canica.fit(canica_data)


def test_canica_square_img(mask_img):
    data, components, rng = _make_canica_test_data(n_subjects=8)

    # We do a large number of inits to be sure to find the good match
    canica = CanICA(
        n_components=4,
        random_state=rng,
        mask=mask_img,
        smoothing_fwhm=0.0,
        n_init=50,
    )
    canica.fit(data)
    maps = get_data(canica.components_img_)
    maps = np.rollaxis(maps, 3, 0)

    # FIXME: This could be done more efficiently, e.g. thanks to hungarian
    # Find pairs of matching components
    # compute the cross-correlation matrix between components
    mask = get_data(mask_img) != 0
    K = np.corrcoef(components[:, mask.ravel()], maps[:, mask])[4:, :4]

    # K should be a permutation matrix, hence its coefficients
    # should all be close to 0 1 or -1
    K_abs = np.abs(K)

    assert np.sum(K_abs > 0.9) == 4

    K_abs[K_abs > 0.9] -= 1

    assert_array_almost_equal(K_abs, 0, 1)


def test_canica_single_subject_smoke():
    """Check that canica runs on a single-subject dataset."""
    data, _, rng = _make_canica_test_data(n_subjects=1)

    canica = CanICA(
        n_components=4, random_state=rng, smoothing_fwhm=0.0, n_init=1
    )

    canica.fit(data[0])


def test_component_sign(mask_img):
    # We should have a heuristic that flips the sign of components in
    # CanICA to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, _, rng = _make_canica_test_data(noisy=True)

    # run CanICA many times (this is known to produce different results)
    canica = CanICA(n_components=4, random_state=rng, mask=mask_img)

    for _ in range(3):
        canica.fit(data)
        for mp in iter_img(canica.components_img_):
            mp = get_data(mp)

            assert -mp.min() <= mp.max()


def test_masker_attributes_with_fit(canica_data, mask_img):
    # Test base module at sub-class

    # Passing mask_img
    canica = CanICA(n_components=3, mask=mask_img, random_state=0)
    canica.fit(canica_data)

    assert canica.mask_img_ == mask_img
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
