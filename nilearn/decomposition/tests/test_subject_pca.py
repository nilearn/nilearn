import numpy as np
from numpy.testing import assert_array_almost_equal
import nibabel

from nilearn.decomposition._subject_pca import _SubjectPCA


def test_single_pca():
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)

    # Create a "multi-subject" dataset
    data = []
    for i in range(8):
        this_data = rng.normal(size=shape)
        # Create fake activation to get non empty mask
        this_data[2:4, 2:4, 2:4, :] += 10
        data.append(nibabel.Nifti1Image(this_data, affine))

    mask_img = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    single_pca = _SubjectPCA(mask=mask_img, n_components=3)

    components = single_pca.fit(data).components_
    assert(components.shape == (3 * 8, 6 * 8 * 10))

    components = single_pca.fit(data[0]).components_
    assert(isinstance(components, np.ndarray))
    assert(components.shape == (3, 6 * 8 * 10))

    cov = components.dot(components.T)
    cov_diag = np.zeros((3, 3))
    for i in range(3):
        cov_diag[i, i] = cov[i, i]
    assert_array_almost_equal(cov - cov_diag, 0)

