"""Tests for :func:`nilearn.plotting.plot_roi`."""

# ruff: noqa: ARG001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image

from nilearn._utils.bids import generate_atlas_look_up_table
from nilearn.conftest import _affine_mni, _img_labels
from nilearn.image.resampling import coord_transform
from nilearn.plotting import plot_roi


def demo_plot_roi(**kwargs):
    """Demo plotting an ROI."""
    data = np.zeros((91, 109, 91))
    # Color a asymmetric rectangle around Broca area.
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(
        x, y, z, np.linalg.inv(_affine_mni())
    )
    data[
        int(x_map) - 5 : int(x_map) + 5,
        int(y_map) - 3 : int(y_map) + 3,
        int(z_map) - 10 : int(z_map) + 10,
    ] = 1
    img = Nifti1Image(data, _affine_mni())
    plot_roi(img, title="Broca's area", **kwargs)


@pytest.mark.slow
@pytest.mark.thread_unsafe
@pytest.mark.parametrize("view_type", ["contours", "continuous"])
@pytest.mark.parametrize("threshold", [0.5, 0.2])
@pytest.mark.parametrize("alpha", [0.7, 0.1])
@pytest.mark.parametrize(
    "display_mode,cut_coords", [("ortho", None), ("z", 3), ("x", [2.0, 10])]
)
def test_plot_roi_view_types(
    matplotlib_pyplot,
    view_type,
    threshold,
    alpha,
    display_mode,
    cut_coords,
):
    """Smoke-test for plot_roi.

    Tests different combinations of parameters `view_type`,
    `threshold`, and `alpha`.
    """
    kwargs = {}
    if view_type == "contours":
        kwargs["linewidth"] = 2.0
    demo_plot_roi(
        view_type=view_type,
        threshold=threshold,
        alpha=alpha,
        display_mode=display_mode,
        cut_coords=cut_coords,
        **kwargs,
    )


@pytest.mark.slow
def test_plot_roi_no_int_64_warning(matplotlib_pyplot, recwarn):
    """Make sure that no int64 warning is thrown."""
    demo_plot_roi()
    for _ in range(len(recwarn)):
        x = recwarn.pop()
        if issubclass(x.category, UserWarning):
            assert "image contains 64-bit ints" not in str(x.message)


def test_plot_roi_view_type_error(matplotlib_pyplot):
    """Test error message for invalid view_type."""
    with pytest.raises(ValueError, match="'view_type' must be one of"):
        demo_plot_roi(view_type="flled")


@pytest.mark.slow
def test_demo_plot_roi_output_file(matplotlib_pyplot, tmp_path):
    """Tests plot_roi file saving capabilities."""
    filename = tmp_path / "test.png"
    out = demo_plot_roi(output_file=filename)
    assert out is None


@pytest.mark.slow
def test_cmap_with_one_level(matplotlib_pyplot, shape_3d_default, affine_eye):
    """Test we can handle cmap with only 1 level.

    Regression test for
    https://github.com/nilearn/nilearn/issues/4255
    """
    array_data = np.zeros(shape_3d_default)
    array_data[0, 1, 1] = 1

    img = Nifti1Image(array_data, affine_eye)

    clust_ids = list(np.unique(img.get_fdata())[1:])

    cmap = plt.get_cmap("tab20", len(clust_ids))

    plot_roi(img, alpha=0.8, colorbar=True, cmap=cmap)


@pytest.mark.slow
def test_cmap_as_lookup_table(img_labels):
    """Test colormap passed as BIDS lookup table."""
    lut = pd.DataFrame(
        {"index": [0, 1], "name": ["foo", "bar"], "color": ["#000", "#fff"]}
    )
    plot_roi(img_labels, cmap=lut)

    lut = pd.DataFrame({"index": [0, 1], "name": ["foo", "bar"]})
    with pytest.warns(
        UserWarning, match="No 'color' column found in the look-up table."
    ):
        plot_roi(img_labels, cmap=lut)


@pytest.mark.slow
@pytest.mark.parametrize("background_label", [None, 0])
def test_cmap_as_lookup_table_with_background(background_label):
    """Ensure that the background color is dropped from lut.

    regression test for https://github.com/nilearn/nilearn/issues/5934
    """
    n_regions = 7

    label_img = _img_labels(n_regions=n_regions)

    lut = generate_atlas_look_up_table(
        index=label_img, background_label=background_label
    )
    color = [
        "#000000",  # 0
        "#781286",  # 1
        "#4682b4",  # 2
        "#00760e",  # 3
        "#c43afa",  # 4
        "#dcf8a4",  # 5
        "#e69422",  # 6
        "#cd3e4e",  # 7
    ]
    lut["color"] = color

    fig = plot_roi(label_img, cmap=lut)

    if background_label is None:
        assert n_regions + 2 == fig._cbar.cmap.N
    else:
        assert n_regions + 1 == fig._cbar.cmap.N
