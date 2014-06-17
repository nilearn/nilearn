# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ..coord_tools import find_xyz_cut_coords

def test_find_cut_coords():
    data = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    data[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    x, y, z = find_xyz_cut_coords(data, mask=np.ones(data.shape, np.bool))
    np.testing.assert_array_equal(
                        (int(round(x)), int(round(y)), int(round(z))),
                                (x_map, y_map, z_map))


