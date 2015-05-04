"""Test CanICA"""

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true, assert_raises, assert_less_equal

import nibabel

from nilearn.decomposition.canica import CanICA


def _make_canica_test_data(rng=None, n_subjects=8):
    shape = (20, 20, 1)
    affine = np.eye(4)
    if rng is None:
        rng = np.random.RandomState(0)

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

    components = np.vstack((component1.ravel(), component2.ravel(),
                            component3.ravel(), component4.ravel()))

    # Create a "multi-subject" dataset
    data = []
    for _ in range(n_subjects):
        this_data = np.dot(rng.normal(size=(40, 4)), components)
        this_data += .01 * rng.normal(size=this_data.shape)
        # Get back into 3D for CanICA
        this_data = np.reshape(this_data, (40,) + shape)
        this_data = np.rollaxis(this_data, 0, 4)
        data.append(nibabel.Nifti1Image(this_data, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape, dtype=np.int8), affine)

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
    K = np.corrcoef(components, maps.reshape(4, 400))[4:, :4]
    # K should be a permutation matrix, hence its coefficients 
    # should all be close to 0 1 or -1
    K_abs = np.abs(K)
    assert_true(np.sum(K_abs > .9) == 4)
    K_abs[K_abs > .9] -= 1
    assert_array_almost_equal(K_abs, 0, 1)

    # Smoke test to make sure an error is raised when no data is passed.
    assert_raises(TypeError, canica.fit)


def test_component_sign():
    # BF for issue #570

    # make data
    data, mask_img, components, rng = _make_canica_test_data(n_subjects=3)

    # fill with -1's
    components[:] = -1

    # have some +1's (so things are not degenerate), but still have more -1's
    # than +1's
    for mp in components:
        mp[rng.randn(*mp.shape) > .8] *= -1
        plus, minus = (mp > 0).sum(), (mp < 0).sum()
        assert_less_equal(plus, minus)

    # fit run CanICA at different times of the day
    canica = CanICA(n_components=4, random_state=rng, mask=mask_img)
    for _ in xrange(3):
        canica.fit(data)
        maps = canica.masker_.inverse_transform(canica.components_).get_data()
        maps = np.rollaxis(maps, 3, 0)
        for mp in maps:
            plus, minus = (mp > 0).sum(), (mp < 0).sum()
            assert_less_equal(minus, plus)

if __name__ == "__main__":
    test_canica_square_img()
