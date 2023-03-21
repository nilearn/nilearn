"""Test CanICA."""

import warnings

import nibabel
import numpy as np
import pytest
from nilearn._utils.testing import write_tmp_imgs
from nilearn.decomposition.canica import CanICA
from nilearn.decomposition.tests.test_multi_pca import _tmp_dir
from nilearn.image import get_data, iter_img
from nilearn.maskers import MultiNiftiMasker
from numpy.testing import assert_array_almost_equal


def _make_data_from_components(
    components, affine, shape, rng=None, n_subjects=8
):
    data = []
    if rng is None:
        rng = np.random.RandomState(0)
    background = -0.01 * rng.normal(size=shape) - 2
    background = background[..., np.newaxis]
    for _ in range(n_subjects):
        this_data = np.dot(rng.normal(size=(40, 4)), components)
        this_data += 0.01 * rng.normal(size=this_data.shape)
        # Get back into 3D for CanICA
        this_data = np.reshape(this_data, (40,) + shape)
        this_data = np.rollaxis(this_data, 0, 4)
        # Put the border of the image to zero, to mimic a brain image
        this_data[:5] = background[:5]
        this_data[-5:] = background[-5:]
        this_data[:, :5] = background[:, :5]
        this_data[:, -5:] = background[:, -5:]
        data.append(nibabel.Nifti1Image(this_data, affine))
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


def _make_canica_test_data(rng=None, n_subjects=8, noisy=True):
    if rng is None:
        rng = np.random.RandomState(0)
    shape = (30, 30, 5)
    affine = np.eye(4)
    components = _make_canica_components(shape)
    if noisy:  # Creating noisy non positive data
        components[rng.randn(*components.shape) > 0.8] *= -2.0

    for mp in components:
        assert mp.max() <= -mp.min()  # Goal met ?

    # Create a "multi-subject" dataset
    data = _make_data_from_components(
        components, affine, shape, rng=rng, n_subjects=n_subjects
    )
    mask = np.ones(shape)
    mask[:5] = 0
    mask[-5:] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0
    mask[..., -2:] = 0
    mask[..., :2] = 0

    mask_img = nibabel.Nifti1Image(mask, affine)
    return data, mask_img, components, rng


def test_canica_square_img():
    data, mask_img, components, rng = _make_canica_test_data()

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

    # Smoke test to make sure an error is raised when no data is passed.
    pytest.raises(TypeError, canica.fit)


def test_canica_single_subject():
    # Check that canica runs on a single-subject dataset
    data, mask_img, components, rng = _make_canica_test_data(n_subjects=1)

    # We do a large number of inits to be sure to find the good match
    canica = CanICA(
        n_components=4, random_state=rng, smoothing_fwhm=0.0, n_init=1
    )
    # This is a smoke test: we just check that things run
    canica.fit(data[0])


def test_component_sign():
    # We should have a heuristic that flips the sign of components in
    # CanICA to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_canica_test_data(
        n_subjects=2, noisy=True
    )

    # run CanICA many times (this is known to produce different results)
    canica = CanICA(n_components=4, random_state=rng, mask=mask_img)
    for _ in range(3):
        canica.fit(data)
        for mp in iter_img(canica.components_img_):
            mp = get_data(mp)
            assert -mp.min() <= mp.max()


def test_threshold_bound():
    # Smoke test to make sure an error is raised when threshold
    # is higher than number of components
    pytest.raises(ValueError, CanICA, n_components=4, threshold=5.0)


def test_percentile_range():
    # Smoke test to test warning in case ignored thresholds
    rng = np.random.RandomState(0)
    edge_case = rng.randint(low=1, high=10)
    data, *_ = _make_canica_test_data()

    # stess thresholding via edge case
    canica = CanICA(n_components=edge_case, threshold=float(edge_case))
    with warnings.catch_warnings(record=True) as warning:
        canica.fit(data)
        # ensure a single warning is raised
        # filter out deprecation warnings
        warning_messages = [
            "obtained a critical threshold" in str(w.message)
            for w in warning
            if not issubclass(w.category, (DeprecationWarning, FutureWarning))
        ]
        assert sum(warning_messages) == 1


def test_masker_attributes_with_fit():
    # Test base module at sub-class
    data, mask_img, components, rng = _make_canica_test_data(n_subjects=3)
    # Passing mask_img
    canica = CanICA(n_components=3, mask=mask_img, random_state=0)
    canica.fit(data)
    assert canica.mask_img_ == mask_img
    assert canica.mask_img_ == canica.masker_.mask_img_
    # Passing masker
    masker = MultiNiftiMasker(mask_img=mask_img)
    canica = CanICA(n_components=3, mask=masker, random_state=0)
    canica.fit(data)
    assert canica.mask_img_ == canica.masker_.mask_img_
    canica = CanICA(mask=mask_img, n_components=3)
    with pytest.raises(
        ValueError,
        match="Object has no components_ attribute. "
        "This is probably because fit has not been called",
    ):
        canica.transform(data)
    # Test if raises an error when empty list of provided.
    with pytest.raises(
        ValueError,
        match="Need one or more Niimg-like objects as input, "
        "an empty list was given.",
    ):
        canica.fit([])
    # Test passing masker arguments to estimator
    canica = CanICA(
        n_components=3,
        target_affine=np.eye(4),
        target_shape=(6, 8, 10),
        mask_strategy="background",
    )
    canica.fit(data)


def test_components_img():
    data, mask_img, _, _ = _make_canica_test_data(n_subjects=3)
    n_components = 3
    canica = CanICA(n_components=n_components, mask=mask_img)
    canica.fit(data)
    components_img = canica.components_img_
    assert isinstance(components_img, nibabel.Nifti1Image)
    check_shape = data[0].shape[:3] + (n_components,)
    assert components_img.shape, check_shape


def test_with_globbing_patterns_with_single_subject():
    # single subject
    data, mask_img, _, _ = _make_canica_test_data(n_subjects=1)
    n_components = 3
    canica = CanICA(n_components=n_components, mask=mask_img)
    with write_tmp_imgs(data[0], create_files=True, use_wildcards=True) as img:
        input_image = _tmp_dir() + img
        canica.fit(input_image)
        components_img = canica.components_img_
        assert isinstance(components_img, nibabel.Nifti1Image)
        # n_components = 3
        check_shape = data[0].shape[:3] + (3,)
        assert components_img.shape, check_shape


def test_with_globbing_patterns_with_multi_subjects():
    # Multi subjects
    data, mask_img, _, _ = _make_canica_test_data(n_subjects=3)
    n_components = 3
    canica = CanICA(n_components=n_components, mask=mask_img)
    with write_tmp_imgs(
        data[0], data[1], data[2], create_files=True, use_wildcards=True
    ) as img:
        input_image = _tmp_dir() + img
        canica.fit(input_image)
        components_img = canica.components_img_
        assert isinstance(components_img, nibabel.Nifti1Image)
        # n_components = 3
        check_shape = data[0].shape[:3] + (3,)
        assert components_img.shape, check_shape


def test_canica_score():
    # Multi subjects
    imgs, mask_img, _, _ = _make_canica_test_data(n_subjects=3)
    n_components = 10
    canica = CanICA(n_components=10, mask=mask_img, random_state=0)
    canica.fit(imgs)

    # One score for all components
    scores = canica.score(imgs, per_component=False)
    assert scores <= 1
    assert 0 <= scores

    # Per component score
    scores = canica.score(imgs, per_component=True)
    assert scores.shape, (n_components,)
    assert np.all(scores <= 1)
    assert np.all(scores >= 0)
