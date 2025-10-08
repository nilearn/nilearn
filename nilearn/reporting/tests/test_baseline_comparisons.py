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
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.image import new_img_like, threshold_img
from nilearn.maskers import SurfaceLabelsMasker, SurfaceMasker
from nilearn.reporting.glm_reporter import _stat_map_to_png
from nilearn.surface.surface import find_surface_clusters


@pytest.mark.timeout(0)
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


def _fs_inflated_sulcal():
    """Load fs average sulcal data on inflated surface."""
    return load_fsaverage_data(mesh_type="inflated")


def _surface_mask_img():
    """Generate surface mask including only high curvature regions."""
    return threshold_img(
        _fs_inflated_sulcal(), 0.5, cluster_threshold=50, two_sided=False
    )


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "mask_img, img",
    (
        [_surface_mask_img(), None],
        [None, _surface_mask_img()],
        [_surface_mask_img(), _fs_inflated_sulcal()],
    ),
)
def test_surface_masker_create_figure_for_report(mask_img, img):
    """Check figure generated in report of SurfaceMasker."""
    masker = SurfaceMasker(mask_img)
    masker.fit(img)
    return masker._create_figure_for_report()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "mask_img, img",
    (
        [_surface_mask_img(), None],
        [None, _fs_inflated_sulcal()],
        [_surface_mask_img(), _fs_inflated_sulcal()],
    ),
)
def test_surface_labels_masker_create_figure_for_report(mask_img, img):
    """Check figure generated in report of SurfaceLabelsMasker."""
    # generate dummy label image
    tmp = _surface_mask_img()
    data = {}
    for hemi in tmp.data.parts:
        _, labels = find_surface_clusters(
            tmp.mesh.parts[hemi], tmp.data.parts[hemi]
        )
        data[hemi] = labels
    labels_img = new_img_like(tmp, data)

    masker = SurfaceLabelsMasker(labels_img, mask_img=mask_img)
    masker.fit(img)
    return masker._create_figure_for_report()
