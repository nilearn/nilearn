import numpy as np
from nose.tools import assert_equal, assert_true
import nibabel
from nilearn.plotting.find_cuts import (find_xyz_cut_coords, find_cut_slices,
                                        _transform_cut_coords, find_parcellation_cut_coords,
                                        find_probabilistic_atlas_cut_coords)


def test_find_cut_coords():
    data = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    data[x_map - 30:x_map + 30, y_map - 3:y_map + 3, z_map - 10:z_map + 10] = 1

    # identity affine
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    x, y, z = find_xyz_cut_coords(img, mask=np.ones(data.shape, np.bool))
    np.testing.assert_allclose((x, y, z),
                               (x_map, y_map, z_map),
                               # Need such a high tolerance for the test to
                               # pass. x, y, z = [49.5, 9.5, 39.5]
                               rtol=6e-2)

    # non-trivial affine
    affine = np.diag([1. / 2, 1 / 3., 1 / 4., 1.])
    img = nibabel.Nifti1Image(data, affine)
    x, y, z = find_xyz_cut_coords(img, mask=np.ones(data.shape, np.bool))
    np.testing.assert_allclose((x, y, z),
                               (x_map / 2., y_map / 3., z_map / 4.),
                               # Need such a high tolerance for the test to
                               # pass. x, y, z = [24.75, 3.17, 9.875]
                               rtol=6e-2)

    # regression test (cf. #473)
    # test case: no data exceeds the activation threshold
    data = np.ones((36, 43, 36))
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    x, y, z = find_xyz_cut_coords(img, activation_threshold=1.1)
    np.testing.assert_array_equal(
        np.array([x, y, z]),
        0.5 * np.array(data.shape).astype(np.float))


def test_find_cut_slices():
    data = np.zeros((50, 50, 50))
    x_map, y_map, z_map = 25, 5, 20
    data[x_map - 15:x_map + 15, y_map - 3:y_map + 3, z_map - 10:z_map + 10] = 1
    img = nibabel.Nifti1Image(data, np.eye(4))
    for n_cuts in (2, 4):
        for direction in 'xz':
            cuts = find_cut_slices(img, direction=direction,
                                   n_cuts=n_cuts, spacing=2)
            # Test that we are indeed getting the right number of cuts
            assert_equal(len(cuts), n_cuts)
            # Test that we are not getting cuts that are separated by
            # less than the minimum spacing that we asked for
            assert_equal(np.diff(cuts).min(), 2)
            # Test that the cuts indeed go through the 'activated' part
            # of the data
            for cut in cuts:
                if direction == 'x':
                    cut_value = data[cut]
                elif direction == 'z':
                    cut_value = data[..., cut]
                assert_equal(cut_value.max(), 1)

    # Now ask more cuts than it is possible to have with a given spacing
    n_cuts = 15
    for direction in 'xz':
        # Only a smoke test
        cuts = find_cut_slices(img, direction=direction,
                               n_cuts=n_cuts, spacing=2)


def test_singleton_ax_dim():
    for axis, direction in enumerate("xyz"):
        shape = [5, 6, 7]
        shape[axis] = 1
        img = nibabel.Nifti1Image(np.ones(shape), np.eye(4))
        find_cut_slices(img, direction=direction)


def test_tranform_cut_coords():
    affine = np.eye(4)

    # test that when n_cuts is 1 we do get an iterable
    for direction in 'xyz':
        assert_true(hasattr(_transform_cut_coords([4], direction, affine),
                            "__iter__"))

    # test that n_cuts after as before function call
    n_cuts = 5
    cut_coords = np.arange(n_cuts)
    for direction in 'xyz':
        assert_equal(len(_transform_cut_coords(cut_coords, direction, affine)),
                     n_cuts)


def test_find_parcellation_atlas_cut_coords():
    data = np.zeros((250, 250, 250))
    x_map_a, y_map_a, z_map_a = 40, 40, 40
    x_map_b, y_map_b, z_map_b = 120, 120, 130
    x_map_c, y_map_c, z_map_c = 200, 200, 200
    data[x_map_a - 10:x_map_a + 10, y_map_a - 10:y_map_a + 10, z_map_a - 20: z_map_a + 20] = 1
    data[x_map_b - 20:x_map_b + 20, y_map_b - 20:y_map_b + 20, z_map_b - 30: z_map_b + 30] = 1
    data[x_map_c - 20:x_map_c + 20, y_map_c - 8:y_map_c + 8, z_map_c - 10: z_map_c + 10] = 2

    # identity affine
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    coords = find_parcellation_cut_coords(img)
    np.testing.assert_allclose((coords[0][0]+.5, coords[0][1]+.5, coords[0][2]+.5),
                               (x_map_b, y_map_b, z_map_b))
    np.testing.assert_allclose((coords[1][0]+.5, coords[1][1]+.5, coords[1][2]+.5),
                               (x_map_c, y_map_c, z_map_c))

    # non-trivial affine
    affine = np.diag([1 / 4., 1 / 5., 1 / 2., 1.])
    img = nibabel.Nifti1Image(data, affine)
    coords = find_parcellation_cut_coords(img)
    np.testing.assert_allclose((coords[0][0], coords[0][1], coords[0][2]),
                               (x_map_b / 4., y_map_b / 5., z_map_b / 2.),
                               rtol=6e-2)
    np.testing.assert_allclose((coords[1][0], coords[1][1], coords[1][2]),
                               (x_map_c / 4., y_map_c / 5., z_map_c / 2.),
                               rtol=6e-2)


def test_find_probabilistic_atlas_cut_coords():
    # make data
    arr1 = np.zeros((200, 200, 200))
    x_map_a, y_map_a, z_map_a = 30, 60, 120
    arr1[x_map_a - 10:x_map_a + 10, y_map_a - 20:y_map_a + 20, z_map_a - 30: z_map_a + 30] = 1

    arr2 = np.zeros((200, 200, 200))
    x_map_b, y_map_b, z_map_b = 40, 50, 60
    arr2[x_map_b - 10:x_map_b + 10, y_map_b - 20:y_map_b + 20, z_map_b - 30: z_map_b + 30] = 1

    data = np.concatenate((arr1[..., np.newaxis], arr2[..., np.newaxis]), axis=3)

    # run test on img with identity affine
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    coords = find_probabilistic_atlas_cut_coords(img)

    np.testing.assert_allclose((coords[0][0], coords[0][1], coords[0][2]),
                               (x_map_a - 0.5, y_map_a - 0.5, z_map_a - 0.5))
    np.testing.assert_allclose((coords[1][0], coords[1][1], coords[1][2]),
                               (x_map_b - 0.5, y_map_b - 0.5, z_map_b - 0.5))

    # non-trivial affine
    affine = np.diag([1 / 5., 1 / 3., 1 / 6., 1.])
    img = nibabel.Nifti1Image(data, affine)
    coords = find_probabilistic_atlas_cut_coords(img)
    np.testing.assert_allclose((coords[0][0], coords[0][1], coords[0][2]),
                               (x_map_a / 5., y_map_a / 3., z_map_a / 6.),
                               # needs high tolerance for the test
                               # to pass.  x, y, z = [9.875, 7.9, 19.75]
                               rtol=6e-2)
    np.testing.assert_allclose((coords[1][0], coords[1][1], coords[1][2]),
                               (x_map_b / 5., y_map_b / 3., z_map_b / 6.),
                               rtol=6e-2)