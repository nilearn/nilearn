import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nilearn.maskers._utils import (
    compute_mean_surface_image,
    concatenate_surface_images,
    deconcatenate_surface_images,
    get_min_max_surface_image,
)
from nilearn.surface import SurfaceImage
from nilearn.surface._testing import (
    assert_polymesh_equal,
    assert_surface_image_equal,
)


def test_compute_mean_surface_image(surf_img_1d, surf_img_2d):
    """Check that mean is properly computed over 'time points'."""
    # one 'time point' image returns same
    img = compute_mean_surface_image(surf_img_1d)

    assert_surface_image_equal(img, surf_img_1d)

    # image with left hemisphere
    # where timepoint 1 has all values == 0
    # and timepoint 2 == 1
    two_time_points_img = surf_img_2d(2)
    two_time_points_img.data.parts["left"][:, 0] = np.zeros(shape=4)
    two_time_points_img.data.parts["left"][:, 1] = np.ones(shape=4)

    img = compute_mean_surface_image(two_time_points_img)

    assert_array_equal(img.data.parts["left"], np.ones(shape=(4,)) * 0.5)
    assert img.shape == (img.mesh.n_vertices,)


def test_get_min_max_surface_image(surf_img_2d):
    """Make sure we get the min and max across hemispheres."""
    img = surf_img_2d()
    img.data.parts["left"][:, 0] = np.zeros(shape=(4))
    img.data.parts["left"][0][0] = 10
    img.data.parts["right"][:, 0] = np.zeros(shape=(5))
    img.data.parts["right"][0][0] = -3.5

    vmin, vmax = get_min_max_surface_image(img)

    assert vmin == -3.5
    assert vmax == 10


def test_concatenate_surface_images(surf_img_2d):
    img = concatenate_surface_images([surf_img_2d(3), surf_img_2d(5)])
    assert img.shape == (9, 8)
    for value in img.data.parts.values():
        assert value.ndim == 2


def test_deconcatenate_surface_images(surf_img_2d):
    input = surf_img_2d(5)
    output = deconcatenate_surface_images(input)

    assert isinstance(output, list)
    assert len(output) == input.shape[1]
    assert all(isinstance(x, SurfaceImage) for x in output)
    for i in range(input.shape[1]):
        assert_polymesh_equal(output[i].mesh, input.mesh)
        assert_array_equal(
            np.squeeze(output[i].data.parts["left"]),
            input.data.parts["left"][..., i],
        )


def test_deconcatenate_surface_images_2d(surf_img_1d, surf_img_2d):
    """Return as is if surface image is 2D."""
    input = surf_img_2d(1)
    output = deconcatenate_surface_images(input)

    assert_surface_image_equal(output[0], input)

    output = deconcatenate_surface_images(surf_img_1d)

    assert_surface_image_equal(output[0], surf_img_1d)


def test_deconcatenate_wrong_input():
    with pytest.raises(TypeError, match="Input must a be SurfaceImage"):
        deconcatenate_surface_images(1)
