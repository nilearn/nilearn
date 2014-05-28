"""Test CanICA"""
import nibabel
import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true
from nilearn.decomposition.canica import CanICA


def test_canica_square_img():
    shape = (20, 20, 1)
    affine = np.eye(4)
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
    for i in range(8):
        this_data = np.dot(rng.normal(size=(40, 4)), components)
        this_data += .01 * rng.normal(size=this_data.shape)
        # Get back into 3D for CanICA
        this_data = np.reshape(this_data, (40,) + shape)
        this_data = np.rollaxis(this_data, 0, 4)
        data.append(nibabel.Nifti1Image(this_data, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape, dtype=np.int8), affine)

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

if __name__ == "__main__":
    test_canica_square_img()
