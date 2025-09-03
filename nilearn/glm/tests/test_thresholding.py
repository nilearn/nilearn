"""Test the thresholding utilities."""

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_equal
from scipy.stats import norm

from nilearn.conftest import _shape_3d_default
from nilearn.glm import (
    cluster_level_inference,
    fdr_threshold,
    threshold_stats_img,
)
from nilearn.glm.thresholding import DEFAULT_Z_THRESHOLD, _compute_hommel_value
from nilearn.image import get_data


def test_fdr(rng):
    n = 100
    x = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    x[:10] = 0.0005
    x = norm.isf(x)
    rng.shuffle(x)

    assert_almost_equal(fdr_threshold(x, 0.1), norm.isf(0.0005))
    assert fdr_threshold(x, 0.001) == np.inf

    # addresses #2879
    n = 10
    pvals = np.linspace(1 / n, 1, n)
    pvals[0] = 0.007

    assert np.isfinite(fdr_threshold(norm.isf(pvals), 0.1))


def test_fdr_error(rng):
    n = 100
    x = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    x[:10] = 0.0005
    x = norm.isf(x)
    rng.shuffle(x)

    match = "alpha should be between 0 and 1"
    with pytest.raises(ValueError, match=match):
        fdr_threshold(x, -0.1)
    with pytest.raises(ValueError, match=match):
        fdr_threshold(x, 1.5)


@pytest.fixture
def data_norm_isf():
    p = np.prod(_shape_3d_default())
    return norm.isf(np.linspace(1.0 / p, 1.0 - 1.0 / p, p)).reshape(
        _shape_3d_default()
    )


@pytest.mark.parametrize("height_control", [None, "fpr", "fdr", "bonferroni"])
def test_threshold_stats_img_warn_threshold_unused(
    data_norm_isf, affine_eye, height_control
):
    """Warn if non default threshold used with height_control != None."""
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    with warnings.catch_warnings(record=True) as warnings_list:
        threshold_stats_img(
            stat_img,
            threshold=2,
            height_control=height_control,
        )
    if height_control is not None:
        assert any("will not be used with" in str(x) for x in warnings_list)


def test_threshold_stats_img_no_height_control(
    data_norm_isf, img_3d_ones_eye, affine_eye
):
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    # excessive cluster forming threshold
    th_map, _ = threshold_stats_img(
        stat_img,
        mask_img=img_3d_ones_eye,
        threshold=100,
        height_control=None,
        cluster_threshold=0,
    )
    vals = get_data(th_map)

    assert np.sum(vals > 0) == 0

    # direct threshold
    th_map, _ = threshold_stats_img(
        stat_img,
        mask_img=img_3d_ones_eye,
        threshold=4.0,
        height_control=None,
        cluster_threshold=0,
    )
    vals = get_data(th_map)

    assert np.sum(vals > 0) == 8

    # without mask
    th_map, _ = threshold_stats_img(
        stat_img, None, threshold=4.0, height_control=None, cluster_threshold=0
    )
    vals = get_data(th_map)

    assert np.sum(vals > 0) == 8

    # without a map
    th_map, threshold = threshold_stats_img(
        None,
        None,
        threshold=DEFAULT_Z_THRESHOLD,
        height_control=None,
        cluster_threshold=0,
    )

    assert threshold == DEFAULT_Z_THRESHOLD
    assert th_map is None


def test_threshold_stats_img(data_norm_isf, img_3d_ones_eye, affine_eye):
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    th_map, _ = threshold_stats_img(
        stat_img,
        mask_img=img_3d_ones_eye,
        alpha=0.001,
        height_control="fpr",
        cluster_threshold=0,
    )
    vals = get_data(th_map)

    assert np.sum(vals > 0) == 8

    # excessive size threshold
    th_map, z_th = threshold_stats_img(
        stat_img,
        mask_img=img_3d_ones_eye,
        alpha=0.001,
        height_control="fpr",
        cluster_threshold=10,
    )
    vals = get_data(th_map)

    assert np.sum(vals > 0) == 0
    assert z_th == norm.isf(0.0005)

    # dr threshold + bonferroni
    for control in ["fdr", "bonferroni"]:
        th_map, _ = threshold_stats_img(
            stat_img,
            mask_img=img_3d_ones_eye,
            alpha=0.05,
            height_control=control,
            cluster_threshold=5,
        )
        vals = get_data(th_map)

        assert np.sum(vals > 0) == 8

    # without a map or mask
    th_map, threshold = threshold_stats_img(
        None, None, alpha=0.05, height_control="fpr", cluster_threshold=0
    )

    assert threshold > 1.64
    assert th_map is None


def test_threshold_stats_img_errors(img_3d_rand_eye):
    with pytest.raises(ValueError, match="'stat_img' cannot be None"):
        threshold_stats_img(None, None, alpha=0.05, height_control="fdr")

    with pytest.raises(ValueError, match="'stat_img' cannot be None"):
        threshold_stats_img(
            None, None, alpha=0.05, height_control="bonferroni"
        )

    with pytest.raises(ValueError, match="'height_control' should be one of"):
        threshold_stats_img(None, None, alpha=0.05, height_control="plop")

    with pytest.raises(
        ValueError, match="should not be a negative value when two_sided=True."
    ):
        threshold_stats_img(
            img_3d_rand_eye, height_control=None, threshold=-2, two_sided=True
        )
    # but this is OK because threshodld is only used
    # when height_control=None
    threshold_stats_img(
        img_3d_rand_eye, height_control="fdr", threshold=-2, two_sided=True
    )


@pytest.mark.parametrize(
    "alpha, expected",
    [
        (1.0e-9, 7),
        (1.0e-7, 6),
        (0.059, 6),
        (0.061, 5),
        (0.249, 5),
        (0.251, 4),
        (0.399, 4),
        (0.401, 3),
        (0.899, 3),
        (0.901, 0),
    ],
)
def test_hommel(alpha, expected):
    """Check that the computation of Hommel value.

    For these, we take the example in  Meijer et al. 2017
    'A shortcut for Hommel's procedure in linearithmic time'
    and check that we obtain the same values.
    https://arxiv.org/abs/1710.08273
    """
    z = norm.isf([1.0e-8, 0.01, 0.08, 0.1, 0.5, 0.7, 0.9])

    assert _compute_hommel_value(z, alpha=alpha) == expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"threshold": DEFAULT_Z_THRESHOLD, "verbose": 1},
            8,
        ),  # standard case (also test verbose)
        ({"threshold": 6}, 0),  # high threshold
        ({"threshold": [3, 6]}, 8),  # list of thresholds
    ],
)
def test_all_resolution_inference(data_norm_isf, affine_eye, kwargs, expected):
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    th_map = cluster_level_inference(stat_img, alpha=0.05, **kwargs)
    vals = get_data(th_map)

    assert np.sum(vals > 0) == expected


def test_all_resolution_inference_with_mask(
    img_3d_ones_eye, affine_eye, data_norm_isf
):
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    th_map = cluster_level_inference(
        stat_img,
        mask_img=img_3d_ones_eye,
        threshold=DEFAULT_Z_THRESHOLD,
        alpha=0.05,
    )
    vals = get_data(th_map)

    assert np.sum(vals > 0) == 8


def test_all_resolution_inference_one_voxel(data_norm_isf, affine_eye):
    data = data_norm_isf
    data[3, 6, 7] = 10
    stat_img = Nifti1Image(data, affine_eye)

    th_map = cluster_level_inference(stat_img, threshold=7, alpha=0.05)
    vals = get_data(th_map)

    assert np.sum(vals > 0) == 1


def test_all_resolution_inference_one_sided(
    data_norm_isf, img_3d_ones_eye, affine_eye
):
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    _, z_th = threshold_stats_img(
        stat_img,
        mask_img=img_3d_ones_eye,
        alpha=0.001,
        height_control="fpr",
        cluster_threshold=10,
        two_sided=False,
    )
    assert_equal(z_th, norm.isf(0.001))


@pytest.mark.parametrize("alpha", [-1, 2])
def test_all_resolution_inference_errors(alpha, data_norm_isf, affine_eye):
    # test aberrant alpha
    data = data_norm_isf
    stat_img = Nifti1Image(data, affine_eye)

    with pytest.raises(ValueError, match="alpha should be between 0 and 1"):
        cluster_level_inference(
            stat_img, threshold=DEFAULT_Z_THRESHOLD, alpha=alpha
        )


@pytest.mark.parametrize("control", ["fdr", "bonferroni"])
def test_all_resolution_inference_height_control(
    control, affine_eye, img_3d_ones_eye, data_norm_isf
):
    # two-side fdr threshold + bonferroni
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    data[0:2, 0:2, 6:8] = -5.0
    stat_img = Nifti1Image(data, affine_eye)

    th_map, _ = threshold_stats_img(
        stat_img,
        mask_img=img_3d_ones_eye,
        alpha=0.05,
        height_control=control,
        cluster_threshold=5,
    )
    vals = get_data(th_map)
    assert_equal(np.sum(vals > 0), 8)
    assert_equal(np.sum(vals < 0), 8)
    th_map, _ = threshold_stats_img(
        stat_img,
        mask_img=img_3d_ones_eye,
        alpha=0.05,
        height_control=control,
        cluster_threshold=5,
        two_sided=False,
    )
    vals = get_data(th_map)

    assert_equal(np.sum(vals > 0), 8)
    assert_equal(np.sum(vals < 0), 0)


@pytest.mark.parametrize("height_control", [None, "bonferroni", "fdr", "fpr"])
def test_threshold_stats_img_surface(surf_img_1d, height_control):
    """Smoke test threshold_stats_img works on surface."""
    threshold_stats_img(
        surf_img_1d,
        height_control=height_control,
        threshold=DEFAULT_Z_THRESHOLD,
    )


def test_threshold_stats_img_surface_with_mask(surf_img_1d, surf_mask_1d):
    """Smoke test threshold_stats_img works on surface with a mask."""
    threshold_stats_img(
        surf_img_1d, height_control="bonferroni", mask_img=surf_mask_1d
    )


def test_threshold_stats_img_surface_output(surf_img_1d):
    """Check output threshold_stats_img surface with no height_control."""
    surf_img_1d.data.parts["left"] = np.asarray([1.0, -1.0, 3.0, 4.0])
    surf_img_1d.data.parts["right"] = np.asarray([2.0, -2.0, 6.0, 8.0, 0.0])

    # two sided
    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, threshold=2
    )

    assert_equal(result.data.parts["left"], np.asarray([0.0, 0.0, 3.0, 4.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, 0.0, 6.0, 8.0, 0.0])
    )

    # one sided positive
    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, threshold=3, two_sided=False
    )

    assert_equal(result.data.parts["left"], np.asarray([0.0, 0.0, 0.0, 4.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, 0.0, 6.0, 8.0, 0.0])
    )

    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, threshold=-0.5, two_sided=False
    )

    # one sided negative
    assert_equal(result.data.parts["left"], np.asarray([0.0, -1.0, 0.0, 0.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, -2.0, 0.0, 0.0, 0.0])
    )


def test_threshold_stats_img_surface_output_threshold_0(surf_img_1d):
    """Check output threshold_stats_img height_control=None, threshold=0."""
    surf_img_1d.data.parts["left"] = np.asarray([1.0, -1.0, 3.0, 4.0])
    surf_img_1d.data.parts["right"] = np.asarray([2.0, -2.0, 6.0, 8.0, 0.0])

    # one sided, with threshold = 0
    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, threshold=0, two_sided=False
    )

    assert_equal(result.data.parts["left"], np.asarray([1.0, 0, 3.0, 4.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([2.0, 0, 6.0, 8.0, 0.0])
    )

    # two sided, with threshold = 0
    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, threshold=0, two_sided=True
    )

    assert_equal(result.data.parts["left"], np.asarray([1.0, -1.0, 3.0, 4.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([2.0, -2.0, 6.0, 8.0, 0.0])
    )


@pytest.mark.parametrize("threshold", [3.0, 2.9, DEFAULT_Z_THRESHOLD])
@pytest.mark.parametrize("height_control", [None, "bonferroni", "fdr", "fpr"])
def test_deprecation_threshold(surf_img_1d, height_control, threshold):
    """Check warning thrown when threshold==old threshold.

    # TODO (nilearn >= 0.15)
    # remove
    """
    with warnings.catch_warnings(record=True) as warning_list:
        threshold_stats_img(
            surf_img_1d, height_control=height_control, threshold=threshold
        )

    n_warnings = len(
        [x for x in warning_list if issubclass(x.category, FutureWarning)]
    )
    if height_control is None and threshold == 3.0:
        assert n_warnings == 1
    else:
        assert n_warnings == 0


@pytest.mark.parametrize("threshold", [3.0, 2.9, DEFAULT_Z_THRESHOLD])
def test_deprecation_threshold_cluster_level_inference(
    threshold, affine_eye, data_norm_isf
):
    """Check cluster_level_inference warns when threshold==old threshold .

    # TODO (nilearn >= 0.15)
    # remove
    """
    data = data_norm_isf
    data[3, 6, 7] = 10
    stat_img = Nifti1Image(data, affine_eye)

    with warnings.catch_warnings(record=True) as warning_list:
        cluster_level_inference(stat_img, threshold=threshold)

    n_warnings = len(
        [x for x in warning_list if issubclass(x.category, FutureWarning)]
    )
    if threshold == 3.0:
        assert n_warnings == 1
    else:
        assert n_warnings == 0
