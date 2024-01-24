"""Utilities for testing maskers."""

from numpy import array_equal

from nilearn._utils.testing import put_last_key_as_first
from nilearn.surface import SurfaceImage


def assert_surf_img_equal(img_1, img_2):
    """Check two surface image objects are equivalent."""
    assert set(img_1.data.keys()) == set(img_2.data.keys())
    for key in img_1.data:
        assert array_equal(img_1.data[key], img_2.data[key])


def drop_hemisphere_surf_img(img, hemisphere="right_hemisphere"):
    """Drop hemisphere part of surface image object."""
    mesh = img.mesh.copy()
    mesh.pop(hemisphere)
    data = img.data.copy()
    data.pop(hemisphere)
    return SurfaceImage(mesh, data)


def flip_surf_img(img):
    """Reorder dictionary keys by putting the last key as the first for mesh \
       and data parts of SurfaceImage object."""
    return SurfaceImage(
        put_last_key_as_first(img.mesh), put_last_key_as_first(img.data)
    )
