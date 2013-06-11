"""Test CanICA"""
import nibabel
import numpy as np
from numpy.testing import assert_array_equal
from nisl.decomposition.old_canica import CanICA


def test_canica_square_img():
    shape = (20, 20)
    rng = np.random.RandomState(0)

    # Create four images with "activated regions"
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
        data.append(this_data)

    canica = CanICA(n_components=4, random_state=rng)
    sparsity = np.infty
    for rs in range(50):
        canica.random_state = np.random.RandomState(rs)
        canica.fit(data)
        maps_ = canica.maps_
        sparsity_ = np.sum(np.abs(maps_), 1).max()
        if sparsity_ < sparsity:
            sparsity = sparsity_
            maps = maps_

    # FIXME: This could be done more efficiently, e.g. thanks to hungarian
    # Find pairs of matching components
    indices = range(4)

    for i in range(4):
        map = np.abs(maps[i]) > np.abs(maps[i]).max() * 0.95
        for j in indices:
            ref_map = components[j].ravel() != 0
            if np.all(map == ref_map):
                indices.remove(j)
                break
        else:
            assert False, "Non matching component"

if __name__ == "__main__":
    test_canica_square_img()
