"""Test CanICA"""
import numpy as np
from numpy.testing import assert_array_equal
from nisl.decomposition import CanICA


def test_canica_square_img():
    rng = np.random.RandomState(0)

    # Create two images with an "activated regions"
    component1 = np.zeros((100, 100))
    component1[:25, :50] = 1
    component1[25:50, :50] = -1

    component2 = np.zeros((100, 100))
    component2[:25, -50:] = 1
    component2[25:50, -50:] = -1

    component3 = np.zeros((100, 100))
    component3[-25:, -50:] = 1
    component3[-50:-25, -50:] = -1

    component4 = np.zeros((100, 100))
    component4[-25:, :50] = 1
    component4[-50:-25, :50] = -1

    components = np.vstack((component1.ravel(), component2.ravel(),
                            component3.ravel(), component4.ravel()))

    # Create a "multi-subject" dataset
    data = []
    for i in range(8):
        this_data = np.dot(rng.normal(size=(40, 4)), components)
        this_data += .01 * rng.normal(size=this_data.shape)
        data.append(this_data)

    canica = CanICA(n_components=4, random_state=rng)
    canica.fit(data)
    maps = canica.maps_

    # FIXME: even with a fixed random state, the ordering of components
    # can change from computer to computer, and break the test.
    assert_array_equal(np.abs(maps[0]) > np.abs(maps[0]).max() * 0.95,
                       component2.ravel() != 0)
    assert_array_equal(np.abs(maps[2]) > np.abs(maps[2]).max() * 0.95,
                       component3.ravel() != 0)
    assert_array_equal(np.abs(maps[1]) > np.abs(maps[1]).max() * 0.95,
                       component1.ravel() != 0)
    assert_array_equal(np.abs(maps[3]) > np.abs(maps[3]).max() * 0.95,
                       component4.ravel() != 0)


if __name__ == "__main__":
    test_canica_square_img()
