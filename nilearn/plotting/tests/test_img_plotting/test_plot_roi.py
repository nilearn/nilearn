"""Tests for :func:`nilearn.plotting.plot_roi`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import _affine_mni
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


@pytest.mark.parametrize("view_type", ["contours", "continuous"])
@pytest.mark.parametrize("black_bg", [True, False])
@pytest.mark.parametrize("threshold", [0.5, 0.2])
@pytest.mark.parametrize("alpha", [0.7, 0.1])
@pytest.mark.parametrize(
    "display_mode,cut_coords", [("ortho", None), ("z", 3), ("x", [2.0, 10])]
)
def test_plot_roi_view_types(
    view_type, black_bg, threshold, alpha, display_mode, cut_coords
):
    """Smoke-test for plot_roi.

    Tests different combinations of parameters `view_type`, `black_bg`,
    `threshold`, and `alpha`.
    """
    kwargs = {}
    if view_type == "contours":
        kwargs["linewidth"] = 2.0
    demo_plot_roi(
        view_type=view_type,
        black_bg=black_bg,
        threshold=threshold,
        alpha=alpha,
        display_mode=display_mode,
        cut_coords=cut_coords,
        **kwargs,
    )
    plt.close()


def test_plot_roi_no_int_64_warning(recwarn):
    """Make sure that no int64 warning is thrown."""
    demo_plot_roi()
    for _ in range(len(recwarn)):
        x = recwarn.pop()
        if issubclass(x.category, UserWarning):
            assert "image contains 64-bit ints" not in str(x.message)


def test_plot_roi_view_type_error():
    """Test error message for invalid view_type."""
    with pytest.raises(ValueError, match="Unknown view type:"):
        demo_plot_roi(view_type="flled")


def test_demo_plot_roi_output_file(tmp_path):
    """Tests plot_roi file saving capabilities."""
    filename = tmp_path / "test.png"
    with filename.open("wb") as fp:
        out = demo_plot_roi(output_file=fp)
    assert out is None


def test_cmap_with_one_level(shape_3d_default, affine_eye):
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
