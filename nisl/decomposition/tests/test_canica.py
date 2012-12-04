"""
Test CanICA
"""
import numpy as np
from nisl.decomposition import CanICA

#def test_canica_square_img():
if 1:
    rng = np.random.RandomState(0)
    # Create two images with an "activated regions"
    component1 = np.zeros((100, 100))
    component1[2:10, 2:10] = 1

    component2 = np.zeros((100, 100))
    component2[-10:-2, -10:-2] = 1
    components = np.vstack((component1.ravel(), component2.ravel()))

    data = []
    # Create a "multi-subject" dataset
    for i in range(3):
        this_data = np.dot(rng.normal((10, 2)), components)
        this_data += .1 * rng.normal(size=this_data.shape)
        data.append(this_data)

    canica = CanICA(n_components=2)
    canica.fit(data)

