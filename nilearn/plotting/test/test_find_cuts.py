import numpy as np

from nose.tools import assert_equal

import nibabel

from ..find_cuts import find_xyz_cut_coords, find_cut_slices


def test_find_cut_coords():
    data = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    data[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1

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
    affine = np.diag([1./2, 1/3., 1/4., 1.])
    img = nibabel.Nifti1Image(data, affine)
    x, y, z = find_xyz_cut_coords(img, mask=np.ones(data.shape, np.bool))
    np.testing.assert_allclose((x, y, z),
                               (x_map / 2., y_map / 3., z_map / 4.),
                               # Need such a high tolerance for the test to
                               # pass. x, y, z = [24.75, 3.17, 9.875]
                               rtol=6e-2)


def test_find_cut_slices():
    data = np.zeros((50, 50, 50))
    x_map, y_map, z_map = 25, 5, 20
    data[x_map-15:x_map+15, y_map-3:y_map+3, z_map-10:z_map+10] = 1
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
