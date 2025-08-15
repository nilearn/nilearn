"""
Test if figure in report output have changed.

See the  maintenance page of our documentation for more information
https://nilearn.github.io/dev/maintenance.html#generating-new-baseline-figures-for-plotting-tests
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from nilearn.datasets import (
    load_fsaverage_data,
    load_mni152_template,
    load_sample_motor_activation_image,
)
from nilearn.glm import threshold_stats_img
from nilearn.reporting.glm_reporter import _stat_map_to_png


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
def test_stat_map_to_png_surface(height_control, two_sided, threshold):
    """Check figures plotting for GLM report for surface data."""
    alpha = 0.05
    cluster_threshold = 0

    surf_img = load_fsaverage_data()

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
    table_details = pd.DataFrame.from_dict(
        table_details,
        orient="index",
    )

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
