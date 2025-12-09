import pytest

from nilearn.plotting.displays._slicers import (
    OrthoSlicer,
    TiledSlicer,
    XSlicer,
    XZSlicer,
    YSlicer,
    YXSlicer,
    YZSlicer,
    ZSlicer,
)


@pytest.mark.parametrize(
    "slicer, cut_coords",
    [
        (XSlicer, [7]),
        (XSlicer, [7, 8]),
        (YSlicer, [8]),
        (YSlicer, [8, 9]),
        (ZSlicer, [9]),
        (ZSlicer, [9, 10]),
        (XZSlicer, [7, 9]),
        (YZSlicer, [8, 9]),
        (YXSlicer, [7, 8]),
        (OrthoSlicer, [7, 8, 9]),
        (TiledSlicer, [7, 8, 9]),
    ],
)
def test_check_cut_coords_in_bounds_error(img_3d_rand_eye, slicer, cut_coords):
    """Test if nilearn.plotting.displays._slicers._check_cut_coords_in_bounds
    raises error when all elements of cut_coords are out of bounds of the
    image for corresponding coordinate.
    """
    # img_3d_rand_eye has bounds (x, y, z):
    # [(0.0, 6.0), (0.0, 7.0), (0.0, 8.0)]
    with pytest.raises(ValueError, match="is out of the bounds of the image"):
        slicer._check_cut_coords_in_bounds(img_3d_rand_eye, cut_coords)


@pytest.mark.parametrize(
    "slicer, cut_coords",
    [
        (XSlicer, [7, 6]),
        (XSlicer, [6, 7, 8]),
        (YSlicer, [7, 8]),
        (YSlicer, [6, 8, 9]),
        (ZSlicer, [8, 9]),
        (ZSlicer, [9, 10, 8]),
        (XZSlicer, [6, 9]),
        (XZSlicer, [7, 8]),
        (YZSlicer, [7, 9]),
        (YZSlicer, [8, 7]),
        (YXSlicer, [6, 8]),
        (YXSlicer, [8, 7]),
        (OrthoSlicer, [6, 8, 9]),
        (OrthoSlicer, [9, 7, 10]),
        (OrthoSlicer, [7, 8, 5]),
        (TiledSlicer, [6, 7, 9]),
        (TiledSlicer, [6, 9, 8]),
        (TiledSlicer, [9, 7, 8]),
    ],
)
def test_cut_coords_out_of_bounds_warning(img_3d_rand_eye, slicer, cut_coords):
    """Test if nilearn.plotting.displays._slicers._check_cut_coords_in_bounds
    warns when at least one but not all of the elements of cut_coords is out of
    bounds of the image for corresponding coordinate.
    """
    # img_3d_rand_eye has bounds:
    # [(0.0, 6.0), (0.0, 7.0), (0.0, 8.0)]
    with pytest.warns(
        UserWarning,
        match=("At least one of the specified cut_coords"),
    ):
        slicer._check_cut_coords_in_bounds(img_3d_rand_eye, cut_coords)
