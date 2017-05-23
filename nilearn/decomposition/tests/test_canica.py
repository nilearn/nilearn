"""Test CanICA"""

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_raises
import nibabel

from nilearn._utils.testing import assert_less_equal
from nilearn.decomposition.canica import CanICA
from nilearn.image import iter_img


def _make_data_from_components(components, affine, shape, rng=None,
                               n_subjects=8):
    data = []
    if rng is None:
        rng = np.random.RandomState(0)
    background = -.01 * rng.normal(size=shape) - 2
    background = background[..., np.newaxis]
    for _ in range(n_subjects):
        this_data = np.dot(rng.normal(size=(40, 4)), components)
        this_data += .01 * rng.normal(size=this_data.shape)
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

    return np.vstack((component1.ravel(), component2.ravel(),
                      component3.ravel(), component4.ravel()))


def _make_canica_test_data(rng=None, n_subjects=8, noisy=False):
    if rng is None:
        rng = np.random.RandomState(0)
    shape = (30, 30, 5)
    affine = np.eye(4)
    components = _make_canica_components(shape)
    if noisy:  # Creating noisy non positive data
        components[rng.randn(*components.shape) > .8] *= -5.

    for mp in components:
        assert_less_equal(mp.max(), -mp.min())  # Goal met ?

    # Create a "multi-subject" dataset
    data = _make_data_from_components(components, affine, shape, rng=rng,
                                      n_subjects=n_subjects)
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
    canica = CanICA(n_components=4, random_state=rng, mask=mask_img,
                    smoothing_fwhm=0., n_init=50)
    canica.fit(data)
    maps = canica.masker_.inverse_transform(canica.components_).get_data()
    maps = np.rollaxis(maps, 3, 0)

    # FIXME: This could be done more efficiently, e.g. thanks to hungarian
    # Find pairs of matching components
    # compute the cross-correlation matrix between components
    mask = mask_img.get_data() != 0
    K = np.corrcoef(components[:, mask.ravel()],
                    maps[:, mask])[4:, :4]
    # K should be a permutation matrix, hence its coefficients
    # should all be close to 0 1 or -1
    K_abs = np.abs(K)
    assert_true(np.sum(K_abs > .9) == 4)
    K_abs[K_abs > .9] -= 1
    assert_array_almost_equal(K_abs, 0, 1)

    # Smoke test to make sure an error is raised when no data is passed.
    assert_raises(TypeError, canica.fit)


def test_canica_single_subject():
    # Check that canica runs on a single-subject dataset
    data, mask_img, components, rng = _make_canica_test_data(n_subjects=1)

    # We do a large number of inits to be sure to find the good match
    canica = CanICA(n_components=4, random_state=rng,
                    smoothing_fwhm=0., n_init=1)
    # This is a smoke test: we just check that things run
    canica.fit(data[0])


def test_component_sign():
    # We should have a heuristic that flips the sign of components in
    # CanICA to have more positive values than negative values, for
    # instance by making sure that the largest value is positive.

    data, mask_img, components, rng = _make_canica_test_data(n_subjects=2,
                                                             noisy=True)

    # run CanICA many times (this is known to produce different results)
    canica = CanICA(n_components=4, random_state=rng, mask=mask_img)
    for _ in range(3):
        canica.fit(data)
        for mp in iter_img(canica.masker_.inverse_transform(
                canica.components_)):
            mp = mp.get_data()
            assert_less_equal(-mp.min(), mp.max())


def test_threshold_bound():
    # Smoke test to make sure an error is raised when threshold
    # is higher than number of components
    assert_raises(ValueError, CanICA, n_components=4, threshold=5.)
