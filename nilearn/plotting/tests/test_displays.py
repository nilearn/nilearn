# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.datasets import load_mni152_template
from nilearn.plotting.displays import (
    BaseAxes,
    LProjector,
    LRProjector,
    LYRProjector,
    LYRZProjector,
    LZRProjector,
    LZRYProjector,
    MosaicSlicer,
    OrthoProjector,
    OrthoSlicer,
    RProjector,
    TiledSlicer,
    XProjector,
    XSlicer,
    XZProjector,
    XZSlicer,
    YProjector,
    YSlicer,
    YXProjector,
    YXSlicer,
    YZProjector,
    YZSlicer,
    ZProjector,
    ZSlicer,
)

SLICER_KEYS = ["ortho", "tiled", "x", "y", "z", "yx", "yz", "mosaic", "xz"]
SLICERS = [
    OrthoSlicer,
    TiledSlicer,
    XSlicer,
    YSlicer,
    ZSlicer,
    YXSlicer,
    YZSlicer,
    MosaicSlicer,
    XZSlicer,
]
PROJECTOR_KEYS = [
    "ortho",
    "xz",
    "yz",
    "yx",
    "lyrz",
    "lyr",
    "lzr",
    "lr",
    "l",
    "r",
]
PROJECTORS = [
    OrthoProjector,
    XZProjector,
    YZProjector,
    YXProjector,
    XProjector,
    YProjector,
    ZProjector,
    LZRYProjector,
    LYRZProjector,
    LYRProjector,
    LZRProjector,
    LRProjector,
    LProjector,
    RProjector,
]


def test_base_axes_exceptions():
    """Tests for exceptions raised by class ``BaseAxes``."""
    axes = BaseAxes(None, "foo", 3)
    # Constructor doesn't raise for invalid direction
    assert axes.direction == "foo"
    assert axes.coord == 3
    with pytest.raises(
        NotImplementedError, match="'transform_to_2d' needs to be"
    ):
        axes.transform_to_2d(None, None)
    with pytest.raises(NotImplementedError, match="'draw_position' should be"):
        axes.draw_position(None, None)
    with pytest.raises(ValueError, match="Invalid value for direction"):
        axes.draw_2d(None, None, None)


def test_cut_axes_exception(affine_eye):
    """Tests for exceptions raised by class ``CutAxes``."""
    from nilearn.plotting.displays import CutAxes

    axes = CutAxes(None, "foo", 2)
    assert axes.direction == "foo"
    assert axes.coord == 2
    with pytest.raises(ValueError, match="Invalid value for direction"):
        axes.transform_to_2d(None, affine_eye)


def test_glass_brain_axes():
    """Tests for class ``GlassBrainAxes``."""
    from nilearn.plotting.displays import GlassBrainAxes

    ax = plt.subplot(111)
    axes = GlassBrainAxes(ax, "r", 2)
    axes._add_markers(np.array([[0, 0, 0]]), "g", [10])
    line_coords = [np.array([[0, 0, 0], [1, 1, 1]])]
    line_values = np.array([1, 0, 6])
    with pytest.raises(
        ValueError, match="If vmax is set to a non-positive number "
    ):
        axes._add_lines(line_coords, line_values, None, vmin=None, vmax=-10)
    axes._add_lines(line_coords, line_values, None, vmin=None, vmax=10)
    with pytest.raises(
        ValueError, match="If vmin is set to a non-negative number "
    ):
        axes._add_lines(line_coords, line_values, None, vmin=10, vmax=None)
    axes._add_lines(line_coords, line_values, None, vmin=-10, vmax=None)
    axes._add_lines(line_coords, line_values, None, vmin=-10, vmax=-5)


def test_get_index_from_direction_exception():
    """Tests that a ValueError is raised when an invalid direction \
       is given to function ``_get_index_from_direction``.
    """
    from nilearn.plotting.displays._axes import _get_index_from_direction

    with pytest.raises(ValueError, match="foo is not a valid direction."):
        _get_index_from_direction("foo")


@pytest.fixture
def img():
    """Image used for testing."""
    return load_mni152_template(resolution=2)


@pytest.fixture
def cut_coords(name):
    """Select appropriate cut coords."""
    if name == "mosaic":
        return 3
    if name in ["yx", "yz", "xz"]:
        return (0,) * 2
    if name in ["lyrz", "lyr", "lzr"]:
        return (0,)
    if name in ["lr", "l"]:
        return (0,) * 4
    return (0,) * 3


@pytest.mark.parametrize(
    "display,name", zip(SLICERS + PROJECTORS, SLICER_KEYS + PROJECTOR_KEYS)
)
def test_display_basics(display, name, img, cut_coords):
    """Basic smoke tests for all displays (slicers + projectors).

    Each object is instantiated, ``add_overlay``, ``title``,
    and ``close`` are then called.
    """
    display = display(cut_coords=cut_coords)
    display.add_overlay(img, cmap=plt.cm.gray)
    display.title(f"display mode is {name}")
    if name != "mosaic":
        assert display.cut_coords == cut_coords
    assert isinstance(display.frame_axes, matplotlib.axes.Axes)
    display.close()


@pytest.mark.parametrize(
    "slicer", [XSlicer, YSlicer, ZSlicer, YXSlicer, YZSlicer, XZSlicer]
)
def test_stacked_slicer(slicer, img, tmp_path):
    """Tests for saving to file with stacked slicers."""
    cut_coords = 3 if slicer in [XSlicer, YSlicer, ZSlicer] else (3, 3)
    slicer = slicer.init_with_figure(img=img, cut_coords=cut_coords)
    slicer.add_overlay(img, cmap=plt.cm.gray)
    # Forcing a layout here, to test the locator code
    slicer.savefig(tmp_path / "out.png")
    slicer.close()


@pytest.mark.parametrize("slicer", [OrthoSlicer, TiledSlicer, MosaicSlicer])
def test_slicer_save_to_file(slicer, img, tmp_path):
    """Tests for saving to file with Ortho/Tiled/Mosaic slicers."""
    cut_coords = None if slicer == MosaicSlicer else (0, 0, 0)
    slicer = slicer.init_with_figure(
        img=img, cut_coords=cut_coords, colorbar=True
    )
    slicer.add_overlay(img, cmap=plt.cm.gray, colorbar=True)
    assert slicer.brain_color == (0.5, 0.5, 0.5)
    assert not slicer.black_bg
    # Forcing a layout here, to test the locator code
    slicer.savefig(tmp_path / "out.png")
    slicer.close()


@pytest.mark.parametrize("cut_coords", [2, 4])
def test_mosaic_slicer_integer_cut_coords(cut_coords, img):
    """Tests for MosaicSlicer with cut_coords provided as an integer."""
    slicer = MosaicSlicer.init_with_figure(img=img, cut_coords=cut_coords)
    slicer.add_overlay(img, cmap=plt.cm.gray, colorbar=True)
    slicer.title("mosaic mode")
    for d in ["x", "y", "z"]:
        assert d in slicer.cut_coords
        assert len(slicer.cut_coords[d]) == cut_coords
    slicer.close()


@pytest.mark.parametrize("cut_coords", [(4, 5, 2), (1, 1, 1)])
def test_mosaic_slicer_tuple_cut_coords(cut_coords, img):
    """Tests for MosaicSlicer with cut_coords provided as a tuple."""
    slicer = MosaicSlicer.init_with_figure(img=img, cut_coords=cut_coords)
    slicer.add_overlay(img, cmap=plt.cm.gray, colorbar=True)
    slicer.title("Showing mosaic mode")
    for i, d in enumerate(["x", "y", "z"]):
        assert len(slicer.cut_coords[d]) == cut_coords[i]
    slicer.close()


@pytest.mark.parametrize("cut_coords", [None, 5, (1, 1, 1)])
def test_mosaic_slicer_img_none_false(cut_coords, img):
    """Tests for MosaicSlicer when img is ``None`` or ``False`` \
       while initializing the figure.
    """
    slicer = MosaicSlicer.init_with_figure(img=None, cut_coords=cut_coords)
    slicer.add_overlay(img, cmap=plt.cm.gray, colorbar=True)
    slicer.close()


@pytest.mark.parametrize("cut_coords", [(5, 4), (1, 2, 3, 4)])
def test_mosaic_slicer_wrong_inputs(cut_coords):
    """Tests that providing wrong inputs raises a ``ValueError``."""
    with pytest.raises(
        ValueError,
        match=(
            "The number cut_coords passed does not "
            "match the display_mode. Mosaic plotting "
            "expects tuple of length 3."
        ),
    ):
        MosaicSlicer.init_with_figure(img=None, cut_coords=cut_coords)
        MosaicSlicer(img=None, cut_coords=cut_coords)


@pytest.fixture
def expected_cuts(cut_coords):
    """Return expected cut with test_demo_mosaic_slicer."""
    if cut_coords == (1, 1, 1):
        return {"x": [-40.0], "y": [-30.0], "z": [-30.0]}
    if cut_coords == 5:
        return {
            "x": [-40.0, -20.0, 0.0, 20.0, 40.0],
            "y": [-30.0, -15.0, 0.0, 15.0, 30.0],
            "z": [-30.0, -3.75, 22.5, 48.75, 75.0],
        }
    return {"x": [10, 20], "y": [30, 40], "z": [15, 16]}


@pytest.mark.parametrize(
    "cut_coords", [(1, 1, 1), 5, {"x": [10, 20], "y": [30, 40], "z": [15, 16]}]
)
def test_demo_mosaic_slicer(cut_coords, img, expected_cuts):
    """Tests for MosaicSlicer with different cut_coords in constructor."""
    slicer = MosaicSlicer(cut_coords=cut_coords)
    slicer.add_overlay(img, cmap=plt.cm.gray)
    assert slicer.cut_coords == expected_cuts
    slicer.close()


@pytest.mark.parametrize("projector", PROJECTORS)
def test_projectors_basic(projector, img, tmp_path):
    """Basic tests for projectors."""
    projector = projector.init_with_figure(img=img)
    projector.add_overlay(img, cmap=plt.cm.gray)
    projector.savefig(tmp_path / "out.png")
    projector.close()


def test_contour_fillings_levels_in_add_contours(img):
    """Tests for method ``add_contours`` of ``OrthoSlicer``."""
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    # levels should be at least 2
    # If single levels are passed then we force upper level to be inf
    oslicer.add_contours(img, filled=True, colors="r", alpha=0.2, levels=[0.0])
    # If two levels are passed, it should be increasing from zero index
    # In this case, we simply omit appending inf
    oslicer.add_contours(
        img, filled=True, colors="b", alpha=0.1, levels=[0.0, 0.2]
    )
    # without passing colors and alpha. In this case, default values are
    # chosen from matplotlib
    oslicer.add_contours(img, filled=True, levels=[0.0, 0.2])

    # levels with only one value
    # vmin argument is not needed but added because of matplotlib 3.8.0rc1 bug
    # see https://github.com/matplotlib/matplotlib/issues/26531
    oslicer.add_contours(img, filled=True, levels=[0.0], vmin=0.0)

    # without passing levels, should work with default levels from
    # matplotlib
    oslicer.add_contours(img, filled=True)
    oslicer.close()


def test_user_given_cmap_with_colorbar(img):
    """Test cmap provided as a string with ``OrthoSlicer``."""
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    oslicer.add_overlay(img, cmap="Paired", colorbar=True)
    oslicer.close()


@pytest.mark.parametrize("display", [OrthoSlicer, LYRZProjector])
def test_data_complete_mask(affine_eye, display):
    """Test for a special case due to matplotlib 2.1.0.

    When the data is completely masked, then we have plotting issues
    See similar issue #9280 reported in matplotlib. This function
    tests the patch added for this particular issue.
    """
    # data is completely masked
    data = np.zeros((10, 20, 30))
    img = Nifti1Image(data, affine_eye)
    n_cuts = 3 if display == OrthoSlicer else 4
    display = display(cut_coords=(0,) * n_cuts)
    display.add_overlay(img)
    display.close()


def test_add_markers_cut_coords_is_none():
    """Tests a special case for ``add_markers`` when ``cut_coords`` are None.

    This case is used when coords are placed on glass brain.
    """
    orthoslicer = OrthoSlicer(cut_coords=(None, None, None))
    orthoslicer.add_markers([(0, 0, 2)])
    orthoslicer.close()


def test_annotations():
    """Tests for ``display.annotate()``.

    In particular, exercise some of the keyword arguments for scale bars.
    """
    orthoslicer = OrthoSlicer(cut_coords=(None, None, None))
    orthoslicer.annotate(size=10, left_right=True, positions=False)
    orthoslicer.annotate(
        size=12,
        left_right=False,
        positions=False,
        scalebar=True,
        scale_size=2.5,
        scale_units="cm",
        scale_loc=3,
        frameon=True,
    )
    orthoslicer.close()


def test_position_annotation_with_decimals():
    """Test of decimals position annotation with precision of 2."""
    orthoslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    orthoslicer.annotate(positions=True, decimals=2)
    orthoslicer.close()


@pytest.mark.parametrize("node_color", ["red", ["red", "blue"]])
def test_add_graph_with_node_color_as_string(node_color):
    """Tests for ``display.add_graph()``."""
    lzry_projector = LZRYProjector(cut_coords=(0, 0, 0, 0))
    matrix = np.array([[0, 3], [3, 0]])
    node_coords = [[-53.60, -62.80, 36.64], [23.87, 0.31, 69.42]]
    lzry_projector.add_graph(matrix, node_coords, node_color=node_color)
    lzry_projector.close()


@pytest.mark.parametrize(
    "threshold,vmin,vmax,expected_results",
    [
        (None, None, None, [[-2, -1, 0], [0, 1, 2]]),
        (0.5, None, None, [[-2, -1, np.nan], [np.nan, 1, 2]]),
        (1, 0, None, [[np.nan, np.nan, np.nan], [np.nan, np.nan, 2]]),
        (1, None, 1, [[-2, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
        (0, 0, 0, [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
    ],
)
def test_threshold(threshold, vmin, vmax, expected_results):
    """Tests for ``OrthoSlicer._threshold``."""
    data = np.array([[-2, -1, 0], [0, 1, 2]], dtype=float)
    assert np.ma.allequal(
        OrthoSlicer._threshold(data, threshold, vmin, vmax),
        np.ma.masked_invalid(expected_results),
    )
