"""Tests for :func:`nilearn.plotting.plot_roi`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.conftest import MNI_AFFINE
from nilearn.image.resampling import coord_transform
from nilearn.plotting import plot_roi


def demo_plot_roi(**kwargs):
    """Demo plotting an ROI."""
    data = np.zeros((91, 109, 91))
    # Color a asymmetric rectangle around Broca area.
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z, np.linalg.inv(MNI_AFFINE))
    data[
        int(x_map) - 5 : int(x_map) + 5,
        int(y_map) - 3 : int(y_map) + 3,
        int(z_map) - 10 : int(z_map) + 10,
    ] = 1
    img = Nifti1Image(data, MNI_AFFINE)
    return plot_roi(img, title="Broca's area", **kwargs)


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
    kwargs = dict()
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


def test_plot_roi_view_type_error():
    """Test error message for invalid view_type."""
    with pytest.raises(ValueError, match="Unknown view type:"):
        demo_plot_roi(view_type="flled")


def test_demo_plot_roi_output_file(tmpdir):
    """Tests plot_roi file saving capabilities."""
    filename = str(tmpdir.join("test.png"))
    with open(filename, "wb") as fp:
        out = demo_plot_roi(output_file=fp)
    assert out is None
