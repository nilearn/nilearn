"""
Test if figure in report output have changed.

See the  maintenance page of our documentation for more information
https://nilearn.github.io/dev/maintenance.html#generating-new-baseline-figures-for-plotting-tests
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image

pytest.importorskip("matplotlib")
import matplotlib as mpl

from nilearn.datasets import (
    load_fsaverage_data,
    load_mni152_template,
    load_sample_motor_activation_image,
)
from nilearn.glm._reporting_utils import _stat_map_to_png
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.tests._testing import block_paradigm
from nilearn.glm.thresholding import threshold_stats_img

pytest.importorskip(
    "matplotlib",
    reason="Matplotlib is not installed; required to run the tests!",
)


@pytest.mark.slow
@pytest.mark.thread_unsafe
@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_type", ["slice", "glass"])
@pytest.mark.parametrize(
    "height_control, two_sided, threshold",
    [
        (None, False, 3),
        (None, False, -3),
        (None, True, 3),
        ("fpr", True, 3),
        ("fpr", False, 3),
    ],
)
@pytest.mark.parametrize("cluster_threshold", [0, 200])
def test_stat_map_to_png_volume(
    plot_type, height_control, two_sided, threshold, cluster_threshold
):
    """Check figures plotting for GLM report."""
    alpha = 0.001

    thresholded_img, threshold = threshold_stats_img(
        stat_img=load_sample_motor_activation_image(),
        threshold=threshold,
        alpha=alpha,
        cluster_threshold=cluster_threshold,
        height_control=height_control,
        two_sided=two_sided,
    )

    table_details = OrderedDict()
    table_details.update({"Threshold Z": np.around(threshold, 3)})
    table_details.update({"two_sided": two_sided})
    table_details.update({"cluster_threshold": cluster_threshold})
    table_details.update({"plot_type": plot_type})
    table_details.update({"height_control": height_control})
    table_details = pd.DataFrame.from_dict(
        table_details,
        orient="index",
    )

    _, fig = _stat_map_to_png(
        stat_img=thresholded_img,
        threshold=threshold,
        bg_img=load_mni152_template(),
        cut_coords=None,
        display_mode="ortho",
        plot_type=plot_type,
        table_details=table_details,
        two_sided=two_sided,
    )

    return fig


@pytest.mark.thread_unsafe
@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "height_control, two_sided, threshold",
    [
        (None, False, 0.5),
        (None, False, -0.5),
        (None, True, 0.5),
        ("bonferroni", True, 3),
        ("bonferroni", False, 3),
    ],
)
@pytest.mark.parametrize("cluster_threshold", [0, 200])
def test_stat_map_to_png_surface(
    height_control, two_sided, threshold, cluster_threshold
):
    """Check figures plotting for GLM report for surface data."""
    alpha = 0.05

    surf_img = load_fsaverage_data(mesh_type="inflated")

    thresholded_img, threshold = threshold_stats_img(
        stat_img=surf_img,
        threshold=threshold,
        alpha=alpha,
        cluster_threshold=cluster_threshold,
        height_control=height_control,
        two_sided=two_sided,
    )

    table_details = OrderedDict()
    table_details.update({"Threshold Z": np.around(threshold, 3)})
    table_details.update({"two_sided": two_sided})
    table_details.update({"height_control": height_control})
    table_details.update({"cluster_threshold": cluster_threshold})
    table_details = pd.DataFrame.from_dict(
        table_details,
        orient="index",
    )

    mpl_rc = mpl.rc_context({"axes.autolimit_mode": "data"})
    with mpl_rc:
        _, fig = _stat_map_to_png(
            stat_img=thresholded_img,
            threshold=threshold,
            bg_img=surf_img,
            cut_coords=None,
            display_mode="ortho",
            plot_type="slice",
            table_details=table_details,
            two_sided=two_sided,
        )

    return fig


@pytest.mark.mpl_image_compare
def test_flm_plot_predicted_signal_and_residuals(
    rng, shape_3d_default, affine_eye, img_3d_ones_eye
):
    """Test plot_predicted_signal_and_residuals with FirstLevelModel."""
    n_frames = 512
    t_r = 1.0
    frame_times = np.linspace(0, (n_frames - 1) * t_r, n_frames)

    dmtx = make_first_level_design_matrix(frame_times, events=block_paradigm())

    images = Nifti1Image(
        dataobj=rng.random((*shape_3d_default, n_frames)), affine=affine_eye
    )

    model = FirstLevelModel(
        t_r=t_r, mask_img=img_3d_ones_eye, minimize_memory=False
    )
    model.fit(images, design_matrices=dmtx)

    _, figs = model.plot_predicted_signal_and_residuals(coords=[(0, 0, 0)])
    return figs[0]


@pytest.mark.mpl_image_compare
def test_slm_plot_predicted_signal_and_residuals(
    rng, affine_eye, shape_3d_default
):
    """Test plot_predicted_signal_and_residuals with SecondLevelModel."""
    images = []
    for _ in range(500):
        data = rng.random(shape_3d_default)
        images.append(Nifti1Image(data, affine_eye))

    design_matrix = pd.DataFrame([1] * len(images), columns=["intercept"])

    model = SecondLevelModel(smoothing_fwhm=8.0, minimize_memory=False)
    model = model.fit(images, design_matrix=design_matrix)
    model.compute_contrast("intercept")

    _, figs = model.plot_predicted_signal_and_residuals(coords=[(0, 0, 0)])
    return figs[0]
