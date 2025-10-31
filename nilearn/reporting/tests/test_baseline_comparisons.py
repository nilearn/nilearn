"""
Test if figure in report output have changed.

See the  maintenance page of our documentation for more information
https://nilearn.github.io/dev/maintenance.html#generating-new-baseline-figures-for-plotting-tests
"""

from collections import OrderedDict

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from nilearn.datasets import (
    load_fsaverage_data,
    load_mni152_gm_mask,
    load_mni152_template,
    load_sample_motor_activation_image,
)
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.image import load_img, math_img, new_img_like, threshold_img
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMapsMasker,
    SurfaceMasker,
)
from nilearn.reporting.glm_reporter import _stat_map_to_png
from nilearn.surface.surface import at_least_2d, find_surface_clusters


@pytest.mark.slow
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
@mpl.rc_context({"axes.autolimit_mode": "data"})
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


def loaded_motor_activation_image():
    """Load motor activation image.

    Needed to standardize image name when used in test parametrization.
    """
    return load_img(load_sample_motor_activation_image())


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "mask_img, img",
    (
        [load_mni152_gm_mask(), None],
        [None, loaded_motor_activation_image()],
        [load_mni152_gm_mask(), loaded_motor_activation_image()],
    ),
)
@pytest.mark.parametrize("src_masker", [NiftiMasker, MultiNiftiMasker])
def test_nifti_masker_create_figure_for_report(src_masker, mask_img, img):
    """Check figure generated in report of NiftiMasker."""
    masker = src_masker(mask_img)
    masker.fit(img)

    displays = masker._create_figure_for_report()
    return displays[0]


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("mask_img", [load_mni152_gm_mask(), None])
@pytest.mark.parametrize("img", [None, loaded_motor_activation_image()])
@pytest.mark.parametrize(
    "src_masker", [NiftiLabelsMasker, MultiNiftiLabelsMasker]
)
def test_nifti_labels_masker_create_figure_for_report(
    src_masker, mask_img, img
):
    """Check figure generated in report of NiftiLabelsMasker."""
    # generate a dummy label image that makes sense for human visualization
    positive_img = threshold_img(
        load_sample_motor_activation_image(),
        3,
        cluster_threshold=300,
        two_sided=False,
    )
    positive_data = positive_img.get_fdata()
    positive_data[positive_data > 0] = 1
    positive_img = new_img_like(positive_img, data=positive_data)

    negative_img = threshold_img(
        load_sample_motor_activation_image(),
        -3,
        cluster_threshold=100,
        two_sided=False,
    )
    negative_data = negative_img.get_fdata()
    negative_data[negative_data < 0] = 2
    negative_img = new_img_like(negative_img, data=negative_data)

    labels_img = math_img("img1 + img2", img1=positive_img, img2=negative_img)

    masker = src_masker(labels_img, mask_img=mask_img)
    masker.fit(img)

    labels_image = masker._reporting_data["labels_image"]

    displays = masker._create_figure_for_report(labels_image)
    return displays[0]


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("mask_img", [load_mni152_gm_mask(), None])
@pytest.mark.parametrize("img", [None, loaded_motor_activation_image()])
@pytest.mark.parametrize("src_masker", [NiftiMapsMasker, MultiNiftiMapsMasker])
def test_nifti_maps_masker_create_figure_for_report(src_masker, mask_img, img):
    """Check figure generated in report of NiftiMapsMasker."""
    # generate dummy maps image
    maps_img = threshold_img(
        load_sample_motor_activation_image(),
        3,
        cluster_threshold=300,
        two_sided=False,
    )

    masker = src_masker(maps_img, mask_img=mask_img)
    masker.fit(img)
    masker._report_content["displayed_maps"] = [0]

    displays = masker._create_figure_for_report()
    return displays[0]


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("mask_img", [load_mni152_gm_mask(), None])
@pytest.mark.parametrize("img", [None, loaded_motor_activation_image()])
def test_nifti_spheres_masker_create_figure_for_report(mask_img, img):
    """Check figure generated in report of NiftiSpheresMasker."""
    masker = NiftiSpheresMasker(seeds=[(0, 0, 0)], mask_img=mask_img)
    masker.fit(img)
    masker._report_content["displayed_maps"] = [0, 1]

    displays = masker._create_figure_for_report()
    return displays[1]


@pytest.mark.mpl_image_compare
def test_nifti_spheres_masker_create_summary_figure_for_report():
    """Check figure with all spheres generated by NiftiSpheresMasker."""
    masker = NiftiSpheresMasker(seeds=[(0, 0, 0), (0, 10, 20), (20, 10, 0)])
    masker.fit()
    masker._report_content["displayed_maps"] = [0]
    displays = masker._create_figure_for_report()
    return displays[0]


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
        [None, _fs_inflated_sulcal()],
        [_surface_mask_img(), _fs_inflated_sulcal()],
    ),
)
def test_surface_masker_create_figure_for_report(mask_img, img):
    """Check figure generated in report of SurfaceMasker."""
    masker = SurfaceMasker(mask_img)
    masker.fit(img)
    return masker._create_figure_for_report()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("mask_img", [_surface_mask_img(), None])
@pytest.mark.parametrize("img", [None, _fs_inflated_sulcal()])
def test_surface_labels_masker_create_figure_for_report(mask_img, img):
    """Check figure generated in report of SurfaceLabelsMasker."""
    # generate dummy labels image
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


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("hemi", ["left", "right"])
@pytest.mark.parametrize("mask_img", [_surface_mask_img(), None])
@pytest.mark.parametrize("img", [None, _fs_inflated_sulcal()])
def test_surface_maps_masker_create_figure_for_report(mask_img, img, hemi):
    """Check figure generated in report of SurfaceMapsMasker."""
    # generate dummy maps image
    # take values main cluster in each hemisphere
    tmp = _surface_mask_img()

    data = {
        "right": np.zeros(tmp.data.parts["right"].shape, dtype=np.float32),
        "left": np.zeros(tmp.data.parts["left"].shape, dtype=np.float32),
    }
    data[hemi] = tmp.data.parts[hemi].astype(np.float32)

    clusters, labels = find_surface_clusters(
        tmp.mesh.parts[hemi], tmp.data.parts[hemi]
    )
    max_size = clusters["size"].max()
    idx_biggest_cluster = clusters["index"][
        clusters["size"] == max_size
    ].to_numpy()

    data[hemi][labels != idx_biggest_cluster] = 0

    maps_imgs = at_least_2d(new_img_like(tmp, data))

    masker = SurfaceMapsMasker(maps_imgs, mask_img=mask_img)
    masker.fit(img)
    masker._report_content["engine"] = "matplotlib"
    return masker._create_figure_for_report(maps_imgs, bg_img=img)
