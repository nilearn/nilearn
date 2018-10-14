import numpy as np
from nose.tools import assert_equal, assert_true, assert_not_equal
import nibabel
from nilearn.plotting.find_cuts import (find_xyz_cut_coords, find_cut_slices,
                                        _transform_cut_coords,
                                        find_parcellation_cut_coords,
                                        find_probabilistic_atlas_cut_coords)
from nilearn._utils.testing import assert_raises_regex, assert_warns
from nilearn.masking import compute_epi_mask


def test_find_cut_coords():
    data = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    data[x_map - 30:x_map + 30, y_map - 3:y_map + 3, z_map - 10:z_map + 10] = 1

    # identity affine
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    mask_img = compute_epi_mask(img)
    x, y, z = find_xyz_cut_coords(img,
                                  mask_img=mask_img)

    np.testing.assert_allclose((x, y, z),
                               (x_map, y_map, z_map),
                               # Need such a high tolerance for the test to
                               # pass. x, y, z = [49.5, 9.5, 39.5]
                               rtol=6e-2)

    # non-trivial affine
    affine = np.diag([1. / 2, 1 / 3., 1 / 4., 1.])
    img = nibabel.Nifti1Image(data, affine)
    mask_img = compute_epi_mask(img)
    x, y, z = find_xyz_cut_coords(img, mask_img=mask_img)
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

    # regression test (cf. #922)
    # pseudo-4D images as input (i.e., X, Y, Z, 1)
    # previously raised "ValueError: too many values to unpack"
    rng = np.random.RandomState(42)
    data_3d = rng.randn(10, 10, 10)
    data_4d = data_3d[..., np.newaxis]
    affine = np.eye(4)
    img_3d = nibabel.Nifti1Image(data_3d, affine)
    img_4d = nibabel.Nifti1Image(data_4d, affine)
    assert_equal(find_xyz_cut_coords(img_3d), find_xyz_cut_coords(img_4d))

    # test passing empty image returns coordinates pointing to AC-PC line
    data = np.zeros((20, 30, 40))
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    cut_coords = find_xyz_cut_coords(img)
    assert_equal(cut_coords, [0.0, 0.0, 0.0])
    cut_coords = assert_warns(UserWarning, find_xyz_cut_coords, img)


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
                    cut_value = data[int(cut)]
                elif direction == 'z':
                    cut_value = data[..., int(cut)]
                assert_equal(cut_value.max(), 1)

    # Now ask more cuts than it is possible to have with a given spacing
    n_cuts = 15
    for direction in 'xz':
        # Only a smoke test
        cuts = find_cut_slices(img, direction=direction,
                               n_cuts=n_cuts, spacing=2)

    # non-diagonal affines
    affine = np.array([[-1., 0., 0., 123.46980286],
                       [0., 0., 1., -94.11079407],
                       [0., -1., 0., 160.694],
                       [0., 0., 0., 1.]])
    img = nibabel.Nifti1Image(data, affine)
    cuts = find_cut_slices(img, direction='z')
    assert_not_equal(np.diff(cuts).min(), 0.)
    affine = np.array([[-2., 0., 0., 123.46980286],
                       [0., 0., 2., -94.11079407],
                       [0., -2., 0., 160.694],
                       [0., 0., 0., 1.]])
    img = nibabel.Nifti1Image(data, affine)
    cuts = find_cut_slices(img, direction='z')
    assert_not_equal(np.diff(cuts).min(), 0.)
    # Rotate it slightly
    angle = np.pi / 180 * 15
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    affine[:2, :2] = rotation_matrix * 2.0
    img = nibabel.Nifti1Image(data, affine)
    cuts = find_cut_slices(img, direction='z')
    assert_not_equal(np.diff(cuts).min(), 0.)


def test_validity_of_ncuts_error_in_find_cut_slices():
    data = np.zeros((50, 50, 50))
    affine = np.eye(4)
    x_map, y_map, z_map = 25, 5, 20
    data[x_map - 15:x_map + 15, y_map - 3:y_map + 3, z_map - 10:z_map + 10] = 1
    img = nibabel.Nifti1Image(data, affine)
    direction = 'z'
    for n_cuts in (0, -2, -10.00034, 0.999999, 0.4, 0.11111111):
        message = ("Image has %d slices in direction %s. Therefore, the number "
                   "of cuts must be between 1 and %d. You provided n_cuts=%s " % (
                       data.shape[0], direction, data.shape[0], n_cuts))
        assert_raises_regex(ValueError,
                            message,
                            find_cut_slices,
                            img, n_cuts=n_cuts)


def test_passing_of_ncuts_in_find_cut_slices():
    data = np.zeros((50, 50, 50))
    affine = np.eye(4)
    x_map, y_map, z_map = 25, 5, 20
    data[x_map - 15:x_map + 15, y_map - 3:y_map + 3, z_map - 10:z_map + 10] = 1
    img = nibabel.Nifti1Image(data, affine)
    # smoke test to check if it rounds the floating point inputs
    for n_cuts in (1, 5., 0.9999999, 2.000000004):
        cut1 = find_cut_slices(img, direction='x', n_cuts=n_cuts)
        cut2 = find_cut_slices(img, direction='x', n_cuts=round(n_cuts))
        np.testing.assert_array_equal(cut1, cut2)


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


def test_find_cuts_empty_mask_no_crash():
    img = nibabel.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
    mask_img = compute_epi_mask(img)
    cut_coords = assert_warns(UserWarning, find_xyz_cut_coords, img,
                              mask_img=mask_img)
    np.testing.assert_array_equal(cut_coords, [.5, .5, .5])


def test_fast_abs_percentile_no_index_error_find_cuts():
    # check that find_cuts functions are safe
    data = np.array([[[1., 2.], [3., 4.]], [[0., 0.], [0., 0.]]])
    img = nibabel.Nifti1Image(data, np.eye(4))
    assert_equal(len(find_xyz_cut_coords(img)), 3)


def test_find_parcellation_cut_coords():
    data = np.zeros((100, 100, 100))
    x_map_a, y_map_a, z_map_a = (10, 10, 10)
    x_map_b, y_map_b, z_map_b = (30, 30, 30)
    x_map_c, y_map_c, z_map_c = (50, 50, 50)
    # Defining 3 parcellations
    data[x_map_a - 10:x_map_a + 10, y_map_a - 10:y_map_a + 10, z_map_a - 10: z_map_a + 10] = 1
    data[x_map_b - 10:x_map_b + 10, y_map_b - 10:y_map_b + 10, z_map_b - 10: z_map_b + 10] = 2
    data[x_map_c - 10:x_map_c + 10, y_map_c - 10:y_map_c + 10, z_map_c - 10: z_map_c + 10] = 3

    # Number of labels
    labels = np.unique(data)
    labels = labels[labels != 0]
    n_labels = len(labels)

    # identity affine
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    # find coordinates with return label names is True
    coords, labels_list = find_parcellation_cut_coords(img,
                                                       return_label_names=True)
    # Check outputs
    assert_equal((n_labels, 3), coords.shape)
    # number of labels in data should equal number of labels list returned
    assert_equal(n_labels, len(labels_list))
    # Labels numbered should match the numbers in returned labels list
    assert_equal(list(labels), labels_list)

    # Match with the number of non-overlapping labels
    np.testing.assert_allclose((coords[0][0], coords[0][1], coords[0][2]),
                               (x_map_a, y_map_a, z_map_a), rtol=6e-2)
    np.testing.assert_allclose((coords[1][0], coords[1][1], coords[1][2]),
                               (x_map_b, y_map_b, z_map_b), rtol=6e-2)
    np.testing.assert_allclose((coords[2][0], coords[2][1], coords[2][2]),
                               (x_map_c, y_map_c, z_map_c), rtol=6e-2)

    # non-trivial affine
    affine = np.diag([1 / 2., 1 / 3., 1 / 4., 1.])
    img = nibabel.Nifti1Image(data, affine)
    coords = find_parcellation_cut_coords(img)
    assert_equal((n_labels, 3), coords.shape)
    np.testing.assert_allclose((coords[0][0], coords[0][1], coords[0][2]),
                               (x_map_a / 2., y_map_a / 3., z_map_a / 4.),
                               rtol=6e-2)
    np.testing.assert_allclose((coords[1][0], coords[1][1], coords[1][2]),
                               (x_map_b / 2., y_map_b / 3., z_map_b / 4.),
                               rtol=6e-2)
    np.testing.assert_allclose((coords[2][0], coords[2][1], coords[2][2]),
                               (x_map_c / 2., y_map_c / 3., z_map_c / 4.),
                               rtol=6e-2)
    # test raises an error with wrong label_hemisphere name with 'lft'
    error_msg = ("Invalid label_hemisphere name:lft. Should be one of "
                 "these 'left' or 'right'.")
    assert_raises_regex(ValueError, error_msg, find_parcellation_cut_coords,
                        labels_img=img, label_hemisphere='lft')


def test_find_probabilistic_atlas_cut_coords():
    # make data
    arr1 = np.zeros((100, 100, 100))
    x_map_a, y_map_a, z_map_a = 30, 40, 50
    arr1[x_map_a - 10:x_map_a + 10, y_map_a - 20:y_map_a + 20, z_map_a - 30: z_map_a + 30] = 1

    arr2 = np.zeros((100, 100, 100))
    x_map_b, y_map_b, z_map_b = 40, 50, 60
    arr2[x_map_b - 10:x_map_b + 10, y_map_b - 20:y_map_b + 20, z_map_b - 30: z_map_b + 30] = 1

    # make data with empty in between non-empty maps to make sure that
    # code does not crash
    arr3 = np.zeros((100, 100, 100))

    data = np.concatenate((arr1[..., np.newaxis], arr3[..., np.newaxis],
                           arr2[..., np.newaxis]), axis=3)

    # Number of maps in time dimension
    n_maps = data.shape[-1]

    # run test on img with identity affine
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine)
    coords = find_probabilistic_atlas_cut_coords(img)

    # Check outputs
    assert_equal((n_maps, 3), coords.shape)

    np.testing.assert_allclose((coords[0][0], coords[0][1], coords[0][2]),
                               (x_map_a, y_map_a, z_map_a), rtol=6e-2)
    np.testing.assert_allclose((coords[2][0], coords[2][1], coords[2][2]),
                               (x_map_b - 0.5, y_map_b - 0.5, z_map_b - 0.5),
                               rtol=6e-2)

    # non-trivial affine
    affine = np.diag([1 / 2., 1 / 3., 1 / 4., 1.])
    img = nibabel.Nifti1Image(data, affine)
    coords = find_probabilistic_atlas_cut_coords(img)
    # Check outputs
    assert_equal((n_maps, 3), coords.shape)
    np.testing.assert_allclose((coords[0][0], coords[0][1], coords[0][2]),
                               (x_map_a / 2., y_map_a / 3., z_map_a / 4.),
                               rtol=6e-2)
    np.testing.assert_allclose((coords[2][0], coords[2][1], coords[2][2]),
                               (x_map_b / 2., y_map_b / 3., z_map_b / 4.),
                               rtol=6e-2)
