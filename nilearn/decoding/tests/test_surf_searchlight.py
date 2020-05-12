"""
Test the surface searchlight
"""
# Author: Sylvain Takerkart
# License: simplified BSD

import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from distutils.version import LooseVersion

import nibabel as nb
from nibabel import gifti

from nilearn.decoding.surf_searchlight import (
    SurfSearchLight, _apply_surfmask_and_get_affinity)


def _create_toy_mesh():
    """Create a basic mesh.
    Returns
    -------
       coords : array, shape(n, 3)
            The xyz coordinates.
       faces : array, shape(m, 3)
    """
    pi = np.pi
    angle_list = [0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3]
    # create list of vertices, starting with the origin
    coords = [[0., 0., 0.]]
    # create empty list of triangles
    faces = []
    for angle_ind, angle in enumerate(angle_list):
        coords.append([np.cos(angle), np.sin(angle), 0.])
        if angle > 0:
            faces.append([0, angle_ind, angle_ind + 1])
    faces.append([0, angle_ind + 1, 1])
    return [np.array(coords), np.array(faces)]


def test__apply_surfmask_and_get_affinity():
    # Create a toy mesh
    mesh = _create_toy_mesh()
    n_vertex = mesh[0].shape[0]
    assert_equal(n_vertex, 7)

    # Generate functional data on this mesh
    n_sample = 2
    func_data = np.ones((n_vertex, n_sample))

    # Prepare inputs to the function to be tested
    mesh_coords = mesh[0]
    giimgs_data = func_data

    # Test with full-mesh mask, small radius
    # each sphere should contain one vertex
    radius = 0.2
    X, A = _apply_surfmask_and_get_affinity(mesh_coords, giimgs_data, radius)
    assert_array_equal(X, func_data.T)
    assert_array_equal(A.toarray(), np.eye(n_vertex))

    # Test with full-mesh mask, large radius
    # each sphere should contain all vertices of the mesh
    radius = 2
    X, A = _apply_surfmask_and_get_affinity(mesh_coords, giimgs_data, radius)
    assert_array_equal(X, func_data.T)
    assert_array_equal(A.toarray(), np.ones([n_vertex, n_vertex]))

    # Create mask covering only partially the mesh (i.e a ROI)
    surfmask = np.zeros(n_vertex)
    surfmask[np.arange(0, n_vertex / 2, dtype=int)] = 1

    # Test with ROI mask, small radius
    # each sphere should contain one vertex
    radius = 0.2
    X, A = _apply_surfmask_and_get_affinity(mesh_coords, giimgs_data, radius,
                                            surfmask=surfmask)
    assert_array_equal(X, func_data.T)
    A_expected = np.eye(n_vertex)
    idx = [np.arange(0, n_vertex / 2, dtype=int)]
    assert_array_equal(A.toarray(), A_expected[idx])

    # Test with ROI mask, large radius
    # each sphere should contain all vertices of the mesh
    radius = 2
    X, A = _apply_surfmask_and_get_affinity(mesh_coords, giimgs_data, radius,
                                            surfmask=surfmask)
    assert_array_equal(X, func_data.T)
    A_expected = np.ones([n_vertex, n_vertex])
    idx = np.arange(0, n_vertex / 2, dtype=int)
    assert_array_equal(A.toarray(), A_expected[idx])


def test_surf_searchlight():
    """Create a toy dataset on a random mesh to run searchlight on"""

    # Generate a mesh
    mesh = _create_toy_mesh()
    n_vertex = mesh[0].shape[0]
    coord_array = gifti.GiftiDataArray(data=mesh[0])
    coord_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET']

    face_array = gifti.GiftiDataArray(data=mesh[1])
    face_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
    gii_mesh = gifti.GiftiImage(darrays=[coord_array, face_array])

    mesh = [gii_mesh.darrays[0].data, gii_mesh.darrays[1].data]

    # Generate a surface mask for a ROI
    data = np.ones(n_vertex, dtype=int)
    if LooseVersion(nb.__version__) > LooseVersion('2.0.2'):
        darray = gifti.GiftiDataArray(data=data)
    else:
        # Avoid a bug in nibabel 1.2.0 where GiftiDataArray were not
        # initialized properly:
        darray = gifti.GiftiDataArray.from_array(data,
                                                 intent='t test')
    gii_mask_tex = gifti.GiftiImage(darrays=[darray])

    surfmask_data = gii_mask_tex.darrays[0].data

    # Create searchlight object
    print(mesh[0].shape)
    searchlight = SurfSearchLight(mesh,
                                  surfmask_data,
                                  process_surfmask_tex=surfmask_data,
                                  cv=4)

    # Generate functional data on this mesh
    rng = np.random.RandomState(0)
    frames = 20
    X = rng.rand(n_vertex, frames)

    # Generate labels for decoding
    y = np.arange(frames, dtype=int) >= (frames // 2)

    # Run searchlight decoding
    scores = searchlight.fit(X, y)

    # Create a toy mesh and dataset to test masking capability
    mesh = _create_toy_mesh()
    n_vertex = mesh[0].shape[0]
    coord_array = gifti.GiftiDataArray(data=mesh[0])
    coord_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET']

    face_array = gifti.GiftiDataArray(data=mesh[1])
    face_array.intent = nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']

    gii_mesh = gifti.GiftiImage(darrays=[coord_array, face_array])

    mesh = [gii_mesh.darrays[0].data, gii_mesh.darrays[1].data]

    # Generate full-mesh mask
    data = np.zeros(n_vertex, dtype=int)
    data[np.arange(n_vertex) > n_vertex / 2] = 1
    if LooseVersion(nb.__version__) > LooseVersion('2.0.2'):
        darray = gifti.GiftiDataArray(data=data)
    else:
        # Avoid a bug in nibabel 1.2.0 where GiftiDataArray were not
        # initialized properly:
        darray = gifti.GiftiDataArray.from_array(data,
                                                 intent='t test')
    gii_mask_tex = gifti.GiftiImage(darrays=[darray])

    surfmask_data = gii_mask_tex.darrays[0].data

    # Create searchlight object
    searchlight = SurfSearchLight(mesh,
                                  surfmask_data,
                                  process_surfmask_tex=surfmask_data,
                                  cv=4)

    # Generate functional data on this mesh
    rng = np.random.RandomState(0)
    frames = 20
    X = rng.rand(n_vertex, frames)

    # Generate labels for decoding
    y = np.arange(frames, dtype=int) >= (frames // 2)

    # Run searchlight decoding
    searchlight.fit(X, y)
    assert len(searchlight.scores_) == len(X)
