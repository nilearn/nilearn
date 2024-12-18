import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_allclose, assert_array_equal

from nilearn.masking import compute_epi_mask
from nilearn.plotting.find_cuts import (
    _transform_cut_coords,
    find_cut_slices,
    find_parcellation_cut_coords,
    find_probabilistic_atlas_cut_coords,
    find_xyz_cut_coords,
)


def test_find_cut_coords(affine_eye):
    data = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    data[
        x_map - 30 : x_map + 30, y_map - 3 : y_map + 3, z_map - 10 : z_map + 10
    ] = 1

    # identity affine
    img = Nifti1Image(data, affine_eye)
    mask_img = compute_epi_mask(img)
    x, y, z = find_xyz_cut_coords(img, mask_img=mask_img)

    assert_allclose(
        (x, y, z),
        (x_map, y_map, z_map),
        # Need such a high tolerance for the test to
        # pass. x, y, z = [49.5, 9.5, 39.5]
        rtol=6e-2,
    )

    # non-trivial affine
    affine = np.diag([1.0 / 2, 1 / 3.0, 1 / 4.0, 1.0])
    img = Nifti1Image(data, affine)
    mask_img = compute_epi_mask(img)
    x, y, z = find_xyz_cut_coords(img, mask_img=mask_img)
    assert_allclose(
        (x, y, z),
        (x_map / 2.0, y_map / 3.0, z_map / 4.0),
        # Need such a high tolerance for the test to
        # pass. x, y, z = [24.75, 3.17, 9.875]
        rtol=6e-2,
    )


def test_no_data_exceeds_activation_threshold(affine_eye):
    """Test when no data exceeds the activation threshold.

    Cut coords should be the center of mass rather than
    the center of the image (10, 10, 10).

    regression test
    https://github.com/nilearn/nilearn/issues/473
    """
    data = np.ones((36, 43, 36))
    img = Nifti1Image(data, affine_eye)
    with pytest.warns(UserWarning, match="All voxels were masked."):
        x, y, z = find_xyz_cut_coords(img, activation_threshold=1.1)
    assert_array_equal([x, y, z], [17.5, 21.0, 17.5])

    data = np.zeros((20, 20, 20))
    data[4:6, 4:6, 4:6] = 1000
    img = Nifti1Image(data, 2 * affine_eye)
    mask_data = np.ones((20, 20, 20), dtype="uint8")
    mask_img = Nifti1Image(mask_data, 2 * affine_eye)
    cut_coords = find_xyz_cut_coords(img, mask_img=mask_img)
    assert_array_equal(cut_coords, [9.0, 9.0, 9.0])


def test_warning_all_voxels_masked(affine_eye):
    """Warning when all values are masked.

    And that the center of mass is returned.
    """
    data = np.zeros((20, 20, 20))
    data[4:6, 4:6, 4:6] = 1000
    img = Nifti1Image(data, affine_eye)

    mask_data = np.ones((20, 20, 20), dtype="uint8")
    mask_data[np.argwhere(data == 1000)] = 0
    mask_img = Nifti1Image(mask_data, affine_eye)

    with pytest.warns(
        UserWarning,
        match=("Could not determine cut coords: All values were masked."),
    ):
        cut_coords = find_xyz_cut_coords(img, mask_img=mask_img)

    assert_array_equal(cut_coords, [4.5, 4.5, 4.5])


def test_warning_all_voxels_masked_thresholding(affine_eye):
    """Warn when all values are masked due to thresholding.

    Also return the center of mass is returned.
    """
    data = np.zeros((20, 20, 20))
    data[4:6, 4:6, 4:6] = 1000
    img = Nifti1Image(data, affine_eye)

    mask_data = np.ones((20, 20, 20), dtype="uint8")

    mask_img = Nifti1Image(mask_data, affine_eye)

    with pytest.warns(
        UserWarning,
        match=(
            "Could not determine cut coords: "
            "All voxels were masked by the thresholding."
        ),
    ):
        cut_coords = find_xyz_cut_coords(
            img, mask_img=mask_img, activation_threshold=10**3
        )
    assert_array_equal(cut_coords, [4.5, 4.5, 4.5])


def test_pseudo_4d_image(rng, shape_3d_default, affine_eye):
    """Check pseudo-4D images as input (i.e., X, Y, Z, 1).

    Previously raised "ValueError: too many values to unpack"
    regression test
    https://github.com/nilearn/nilearn/issues/922
    """
    data_3d = rng.standard_normal(size=shape_3d_default)
    data_4d = data_3d[..., np.newaxis]
    img_3d = Nifti1Image(data_3d, affine_eye)
    img_4d = Nifti1Image(data_4d, affine_eye)
    assert find_xyz_cut_coords(img_3d) == find_xyz_cut_coords(img_4d)


def test_empty_image_ac_pc_line(img_3d_zeros_eye):
    """Pass empty image returns coordinates pointing to AC-PC line."""
    with pytest.warns(UserWarning, match="Given img is empty."):
        cut_coords = find_xyz_cut_coords(img_3d_zeros_eye)
    assert cut_coords == [0.0, 0.0, 0.0]


def test_find_cut_slices(affine_eye):
    data = np.zeros((50, 50, 50))
    x_map, y_map, z_map = 25, 5, 20
    data[
        x_map - 15 : x_map + 15, y_map - 3 : y_map + 3, z_map - 10 : z_map + 10
    ] = 1
    img = Nifti1Image(data, affine_eye)
    for n_cuts in (2, 4):
        for direction in "xz":
            cuts = find_cut_slices(
                img, direction=direction, n_cuts=n_cuts, spacing=2
            )
            # Test that we are indeed getting the right number of cuts
            assert len(cuts) == n_cuts
            # Test that we are not getting cuts that are separated by
            # less than the minimum spacing that we asked for
            assert np.diff(cuts).min() == 2
            # Test that the cuts indeed go through the 'activated' part
            # of the data
            for cut in cuts:
                if direction == "x":
                    cut_value = data[int(cut)]
                elif direction == "z":
                    cut_value = data[..., int(cut)]
                assert cut_value.max() == 1

    # Now ask more cuts than it is possible to have with a given spacing
    n_cuts = 15
    for direction in "xz":
        # Only a smoke test
        cuts = find_cut_slices(
            img, direction=direction, n_cuts=n_cuts, spacing=2
        )

    # non-diagonal affines
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 123.46980286],
            [0.0, 0.0, 1.0, -94.11079407],
            [0.0, -1.0, 0.0, 160.694],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = Nifti1Image(data, affine)
    cuts = find_cut_slices(img, direction="z")
    assert np.diff(cuts).min() != 0.0
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 123.46980286],
            [0.0, 0.0, 2.0, -94.11079407],
            [0.0, -2.0, 0.0, 160.694],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = Nifti1Image(data, affine)
    cuts = find_cut_slices(img, direction="z")
    assert np.diff(cuts).min() != 0.0
    # Rotate it slightly
    angle = np.pi / 180 * 15
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    affine[:2, :2] = rotation_matrix * 2.0
    img = Nifti1Image(data, affine)
    cuts = find_cut_slices(img, direction="z")
    assert np.diff(cuts).min() != 0.0


def test_validity_of_ncuts_error_in_find_cut_slices(affine_eye):
    data = np.zeros((50, 50, 50))
    x_map, y_map, z_map = 25, 5, 20
    data[
        x_map - 15 : x_map + 15, y_map - 3 : y_map + 3, z_map - 10 : z_map + 10
    ] = 1
    img = Nifti1Image(data, affine_eye)
    direction = "z"
    for n_cuts in (0, -2, -10.00034, 0.999999, 0.4, 0.11111111):
        message = (
            f"Image has {data.shape[0]} slices in direction {direction}. "
            "Therefore, the number of cuts "
            f"must be between 1 and {data.shape[0]}. "
            f"You provided n_cuts={n_cuts}."
        )
        with pytest.raises(ValueError, match=message):
            find_cut_slices(img, n_cuts=n_cuts)


def test_passing_of_ncuts_in_find_cut_slices(affine_eye):
    data = np.zeros((50, 50, 50))
    x_map, y_map, z_map = 25, 5, 20
    data[
        x_map - 15 : x_map + 15, y_map - 3 : y_map + 3, z_map - 10 : z_map + 10
    ] = 1
    img = Nifti1Image(data, affine_eye)
    # smoke test to check if it rounds the floating point inputs
    for n_cuts in (1, 5.0, 0.9999999, 2.000000004):
        cut1 = find_cut_slices(img, direction="x", n_cuts=n_cuts)
        cut2 = find_cut_slices(img, direction="x", n_cuts=round(n_cuts))
        assert_array_equal(cut1, cut2)


def test_singleton_ax_dim(affine_eye):
    for axis, direction in enumerate("xyz"):
        shape = [5, 6, 7]
        shape[axis] = 1
        img = Nifti1Image(np.ones(shape), affine_eye)
        find_cut_slices(img, direction=direction)


def test_tranform_cut_coords(affine_eye):
    # test that when n_cuts is 1 we do get an iterable
    for direction in "xyz":
        assert hasattr(
            _transform_cut_coords([4], direction, affine_eye), "__iter__"
        )

    # test that n_cuts after as before function call
    n_cuts = 5
    cut_coords = np.arange(n_cuts)
    for direction in "xyz":
        assert (
            len(_transform_cut_coords(cut_coords, direction, affine_eye))
            == n_cuts
        )


def test_find_cuts_empty_mask_no_crash(affine_eye):
    img = Nifti1Image(np.ones((2, 2, 2)), affine_eye)
    mask_img = compute_epi_mask(img)
    with pytest.warns(UserWarning):
        cut_coords = find_xyz_cut_coords(img, mask_img=mask_img)
    assert_array_equal(cut_coords, [0.5, 0.5, 0.5])


def test_fast_abs_percentile_no_index_error_find_cuts(affine_eye):
    # check that find_cuts functions are safe
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
    img = Nifti1Image(data, affine_eye)
    assert len(find_xyz_cut_coords(img)) == 3


def test_find_parcellation_cut_coords(affine_eye):
    data = np.zeros((100, 100, 100))
    x_map_a, y_map_a, z_map_a = (10, 10, 10)
    x_map_b, y_map_b, z_map_b = (30, 30, 30)
    x_map_c, y_map_c, z_map_c = (50, 50, 50)
    # Defining 3 parcellations
    data[
        x_map_a - 10 : x_map_a + 10,
        y_map_a - 10 : y_map_a + 10,
        z_map_a - 10 : z_map_a + 10,
    ] = 2301
    data[
        x_map_b - 10 : x_map_b + 10,
        y_map_b - 10 : y_map_b + 10,
        z_map_b - 10 : z_map_b + 10,
    ] = 4001
    data[
        x_map_c - 10 : x_map_c + 10,
        y_map_c - 10 : y_map_c + 10,
        z_map_c - 10 : z_map_c + 10,
    ] = 6201

    # Number of labels
    labels = np.unique(data)
    labels = labels[labels != 0]
    n_labels = len(labels)

    # identity affine
    img = Nifti1Image(data, affine_eye)
    # find coordinates with return label names is True
    coords, labels_list = find_parcellation_cut_coords(
        img, return_label_names=True
    )
    # Check outputs
    assert (n_labels, 3) == coords.shape
    # number of labels in data should equal number of labels list returned
    assert n_labels == len(labels_list)
    # Labels numbered should match the numbers in returned labels list
    assert list(labels) == labels_list

    # Match with the number of non-overlapping labels
    assert_allclose(
        (coords[0][0], coords[0][1], coords[0][2]),
        (x_map_a, y_map_a, z_map_a),
        rtol=6e-2,
    )
    assert_allclose(
        (coords[1][0], coords[1][1], coords[1][2]),
        (x_map_b, y_map_b, z_map_b),
        rtol=6e-2,
    )
    assert_allclose(
        (coords[2][0], coords[2][1], coords[2][2]),
        (x_map_c, y_map_c, z_map_c),
        rtol=6e-2,
    )

    # non-trivial affine
    affine = np.diag([1 / 2.0, 1 / 3.0, 1 / 4.0, 1.0])
    img = Nifti1Image(data, affine)
    coords = find_parcellation_cut_coords(img)
    assert (n_labels, 3) == coords.shape
    assert_allclose(
        (coords[0][0], coords[0][1], coords[0][2]),
        (x_map_a / 2.0, y_map_a / 3.0, z_map_a / 4.0),
        rtol=6e-2,
    )
    assert_allclose(
        (coords[1][0], coords[1][1], coords[1][2]),
        (x_map_b / 2.0, y_map_b / 3.0, z_map_b / 4.0),
        rtol=6e-2,
    )
    assert_allclose(
        (coords[2][0], coords[2][1], coords[2][2]),
        (x_map_c / 2.0, y_map_c / 3.0, z_map_c / 4.0),
        rtol=6e-2,
    )
    # test raises an error with wrong label_hemisphere name with 'lft'
    error_msg = (
        "Invalid label_hemisphere name:lft.\nShould be one of "
        "these 'left' or 'right'."
    )
    with pytest.raises(ValueError, match=error_msg):
        find_parcellation_cut_coords(labels_img=img, label_hemisphere="lft")


def test_find_parcellation_cut_coords_hemispheres(affine_mni):
    # Create a mock labels_img object
    data = np.zeros((10, 10, 10))
    data[2:5, 2:5, 2:5] = 1  # left hemisphere
    labels_img = Nifti1Image(data, affine_mni)

    # Test when label_hemisphere is "left"
    coords, labels = find_parcellation_cut_coords(
        labels_img, return_label_names=True, label_hemisphere="left"
    )
    assert len(coords) == 1
    assert labels == [1]

    # Test when label_hemisphere is "right"
    coords, labels = find_parcellation_cut_coords(
        labels_img, return_label_names=True, label_hemisphere="right"
    )
    assert len(coords) == 1
    assert labels == [1]


def test_find_probabilistic_atlas_cut_coords(affine_eye):
    # make data
    arr1 = np.zeros((100, 100, 100))
    x_map_a, y_map_a, z_map_a = 30, 40, 50
    arr1[
        x_map_a - 10 : x_map_a + 10,
        y_map_a - 20 : y_map_a + 20,
        z_map_a - 30 : z_map_a + 30,
    ] = 1

    arr2 = np.zeros((100, 100, 100))
    x_map_b, y_map_b, z_map_b = 40, 50, 60
    arr2[
        x_map_b - 10 : x_map_b + 10,
        y_map_b - 20 : y_map_b + 20,
        z_map_b - 30 : z_map_b + 30,
    ] = 1

    # make data with empty in between non-empty maps to make sure that
    # code does not crash
    arr3 = np.zeros((100, 100, 100))

    data = np.concatenate(
        (arr1[..., np.newaxis], arr3[..., np.newaxis], arr2[..., np.newaxis]),
        axis=3,
    )

    # Number of maps in time dimension
    n_maps = data.shape[-1]

    # run test on img with identity affine
    img = Nifti1Image(data, affine_eye)
    coords = find_probabilistic_atlas_cut_coords(img)

    # Check outputs
    assert (n_maps, 3) == coords.shape

    assert_allclose(
        (coords[0][0], coords[0][1], coords[0][2]),
        (x_map_a, y_map_a, z_map_a),
        rtol=6e-2,
    )
    assert_allclose(
        (coords[2][0], coords[2][1], coords[2][2]),
        (x_map_b - 0.5, y_map_b - 0.5, z_map_b - 0.5),
        rtol=6e-2,
    )

    # non-trivial affine
    affine = np.diag([1 / 2.0, 1 / 3.0, 1 / 4.0, 1.0])
    img = Nifti1Image(data, affine)
    coords = find_probabilistic_atlas_cut_coords(img)
    # Check outputs
    assert (n_maps, 3) == coords.shape
    assert_allclose(
        (coords[0][0], coords[0][1], coords[0][2]),
        (x_map_a / 2.0, y_map_a / 3.0, z_map_a / 4.0),
        rtol=6e-2,
    )
    assert_allclose(
        (coords[2][0], coords[2][1], coords[2][2]),
        (x_map_b / 2.0, y_map_b / 3.0, z_map_b / 4.0),
        rtol=6e-2,
    )
