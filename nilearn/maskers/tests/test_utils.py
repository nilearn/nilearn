import numpy as np
from numpy.testing import assert_array_equal

from nilearn.maskers._utils import (
    compute_mean_surface_image,
    concatenate_surface_images,
    get_min_max_surface_image,
)


def test_compute_mean_surface_image(surf_img, assert_surf_img_equal):
    """Check that mean is properly computed over 'time points'."""
    # one 'time point' image returns same
    input_img = surf_img()
    img = compute_mean_surface_image(input_img)

    assert_surf_img_equal(img, input_img)

    # image with left hemisphere
    # where timepoint 1 has all values == 0
    # and timepoint 2 == 1
    two_time_points_img = surf_img((2,))
    two_time_points_img.data.parts["left"][0] = np.zeros(shape=(1, 4))
    two_time_points_img.data.parts["left"][1] = np.ones(shape=(1, 4))

    img = compute_mean_surface_image(two_time_points_img)

    assert_array_equal(img.data.parts["left"], np.ones(shape=(1, 4)) * 0.5)
    assert img.shape == (1, img.mesh.n_vertices)


def test_get_min_max_surface_image(surf_img):
    """Make sure we get the min and max across hemispheres."""
    img = surf_img()
    img.data.parts["left"][0] = np.zeros(shape=(4))
    img.data.parts["left"][0][0] = 10
    img.data.parts["right"][0] = np.zeros(shape=(5))
    img.data.parts["right"][0][0] = -3.5

    vmin, vmax = get_min_max_surface_image(img)

    assert vmin == -3.5
    assert vmax == 10


def test_concatenate_surface_images(surf_img):
    img = concatenate_surface_images([surf_img(3), surf_img(5)])
    assert img.shape == (9, 8)
    for value in img.data.parts.values():
        assert value.ndim == 2
