"""Test the thresholding utilities."""

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_equal
from scipy.stats import norm

from nilearn.datasets import (
    load_fsaverage_data,
    load_sample_motor_activation_image,
)
from nilearn.exceptions import DimensionError
from nilearn.glm import (
    cluster_level_inference,
    fdr_threshold,
    threshold_stats_img,
)
from nilearn.glm.thresholding import DEFAULT_Z_THRESHOLD, _compute_hommel_value
from nilearn.image import get_data, new_img_like
from nilearn.surface.surface import PolyData
from nilearn.surface.surface import get_data as get_surf_data


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


def _data_norm_isf(shape):
    p = np.prod(shape)
    return norm.isf(np.linspace(1.0 / p, 1.0 - 1.0 / p, p)).reshape(shape)


@pytest.fixture
def data_norm_isf(shape_3d_default):
    return _data_norm_isf(shape_3d_default)


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
        assert any("is not used with" in str(x) for x in warnings_list)


@pytest.mark.slow
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


def test_threshold_stats_img_error_height_control(
    data_norm_isf, img_3d_ones_eye, affine_eye
):
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    with pytest.raises(ValueError, match="must be one of"):
        threshold_stats_img(
            stat_img,
            mask_img=img_3d_ones_eye,
            height_control="knights_of_ni",
        )


def test_threshold_stats_img_error_cluster_threshold(
    data_norm_isf, img_3d_ones_eye, affine_eye
):
    """Raise error for invalid cluster_threshold."""
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    with pytest.raises(ValueError, match="'cluster_threshold' must be > 0"):
        threshold_stats_img(
            stat_img, mask_img=img_3d_ones_eye, cluster_threshold=-10
        )


@pytest.mark.slow
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

    with pytest.raises(ValueError, match="'height_control' must be one of"):
        threshold_stats_img(None, None, alpha=0.05, height_control="plop")

    with pytest.raises(
        ValueError,
        match=r"should not be a negative value when two_sided=True.",
    ):
        threshold_stats_img(
            img_3d_rand_eye, height_control=None, threshold=-2, two_sided=True
        )
    # but this is OK because threshold is only used
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "kwargs, expected, expected_n_unique_values",
    [
        (
            {"threshold": DEFAULT_Z_THRESHOLD, "verbose": 1},
            8,
            2,
        ),  # standard case (also test verbose)
        ({"threshold": 6}, 0, 1),  # high threshold
        ({"threshold": [3, 6]}, 8, 2),  # list of thresholds
    ],
)
def test_all_resolution_inference(
    data_norm_isf, affine_eye, kwargs, expected, expected_n_unique_values
):
    data = data_norm_isf
    data[2:4, 5:7, 6:8] = 5.0
    stat_img = Nifti1Image(data, affine_eye)

    th_map = cluster_level_inference(stat_img, alpha=0.05, **kwargs)
    vals = get_data(th_map)

    assert np.sum(vals > 0) == expected
    # only one unique non zero value: one per cluster
    assert len(np.unique(vals)) == expected_n_unique_values


@pytest.mark.parametrize(
    "kwargs, expected_left, expected_right, expected_n_unique_values",
    [
        (
            {"threshold": DEFAULT_Z_THRESHOLD, "verbose": 1},
            2,
            3,
            2,
        ),  # standard case (also test verbose)
        ({"threshold": 6}, 0, 0, 1),  # high threshold
        ({"threshold": [3, 6]}, 2, 3, 2),  # list of thresholds
    ],
)
def test_all_resolution_inference_surface(
    surf_img_1d,
    kwargs,
    expected_left,
    expected_right,
    expected_n_unique_values,
):
    """Check cluster_level_inference that runs on each hemisphere."""
    data_left = _data_norm_isf(surf_img_1d.data.parts["left"].shape)
    data_left[2:4] = 5.0
    data_right = _data_norm_isf(surf_img_1d.data.parts["right"].shape)
    data_right[2:5] = 5.0

    stat_img = new_img_like(
        surf_img_1d, PolyData(left=data_left, right=data_right)
    )

    th_map = cluster_level_inference(stat_img, alpha=0.05, **kwargs)

    assert np.sum(th_map.data.parts["left"] > 0) == expected_left
    # only one unique non zero value: one per cluster
    assert (
        len(np.unique(th_map.data.parts["left"])) == expected_n_unique_values
    )

    assert np.sum(th_map.data.parts["right"] > 0) == expected_right
    assert (
        len(np.unique(th_map.data.parts["right"])) == expected_n_unique_values
    )


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


@pytest.mark.slow
@pytest.mark.parametrize(
    "threshold, expected_n_unique_values",
    [
        (2.5, 3),
        ([2.5, 3.5], 6),
        ([2.5, 3.0, 3.5], 8),
    ],
)
def test_cluster_level_inference_realistic_data(
    threshold, expected_n_unique_values
):
    """Check cluster_level_inference on realistic data."""
    stat_img = load_sample_motor_activation_image()
    th_map = cluster_level_inference(stat_img, threshold=threshold)
    vals = th_map.get_fdata()
    assert len(np.unique(vals)) == expected_n_unique_values


def test_all_resolution_inference_surface_mask(surf_img_1d):
    """Check cluster_level_inference that runs on each hemisphere.

    Here mask excludes the right hemisphere.
    """
    data_left = _data_norm_isf(surf_img_1d.data.parts["left"].shape)
    data_left[2:4] = 5.0
    data_right = _data_norm_isf(surf_img_1d.data.parts["right"].shape)
    data_right[2:5] = 5.0
    stat_img = new_img_like(
        surf_img_1d, {"left": data_left, "right": data_right}
    )

    mask_left = np.ones(surf_img_1d.data.parts["left"].shape)
    mask_right = np.zeros(surf_img_1d.data.parts["right"].shape)
    mask_img = new_img_like(
        surf_img_1d, data={"left": mask_left, "right": mask_right}
    )

    th_map = cluster_level_inference(
        stat_img,
        mask_img=mask_img,
        threshold=DEFAULT_Z_THRESHOLD,
        alpha=0.05,
    )

    assert np.sum(th_map.data.parts["left"] > 0) == 2
    assert np.sum(th_map.data.parts["right"] > 0) == 0


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
def test_all_resolution_inference_alpha_errors(
    alpha, data_norm_isf, affine_eye
):
    """Test aberrant alpha."""
    data = data_norm_isf
    stat_img = Nifti1Image(data, affine_eye)

    with pytest.raises(ValueError, match="alpha should be between 0 and 1"):
        cluster_level_inference(
            stat_img, threshold=DEFAULT_Z_THRESHOLD, alpha=alpha
        )


@pytest.mark.parametrize("threshold", [-1, [-1, 2]])
def test_all_resolution_inference_threshold_errors(
    data_norm_isf, affine_eye, threshold
):
    """Test aberrant threshold."""
    data = data_norm_isf
    stat_img = Nifti1Image(data, affine_eye)

    with pytest.raises(
        ValueError,
        match=("'threshold' cannot be negative or contain negative values"),
    ):
        cluster_level_inference(stat_img, threshold=threshold)


def test_all_resolution_inference_shape_errors(img_4d_rand_eye, surf_img_2d):
    """Raise error with 4D image or SurfaceImage with n_samples>1."""
    with pytest.raises(
        DimensionError,
        match=("Input data has incompatible dimensionality"),
    ):
        cluster_level_inference(img_4d_rand_eye, threshold=0.5)

    with pytest.raises(
        ValueError,
        match=("Data for each part of img should be 1D"),
    ):
        cluster_level_inference(surf_img_2d(2), threshold=0.5)


@pytest.mark.slow
@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("control", ["fdr", "bonferroni"])
def test_all_resolution_inference_height_control(
    control, affine_eye, img_3d_ones_eye, data_norm_isf, two_sided
):
    """Test FDR threshold/bonferroni with one/two sided."""
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
        two_sided=two_sided,
    )

    vals = get_data(th_map)
    assert_equal(np.sum(vals > 0), 8)
    if two_sided:
        assert_equal(np.sum(vals < 0), 8)
    else:
        assert_equal(np.sum(vals < 0), 0)


@pytest.mark.parametrize("height_control", [None, "bonferroni", "fdr", "fpr"])
def test_threshold_stats_img_surface(surf_img_1d, height_control):
    """Smoke test threshold_stats_img works on surface."""
    with warnings.catch_warnings(record=True) as warning_list:
        threshold_stats_img(
            surf_img_1d,
            height_control=height_control,
            threshold=DEFAULT_Z_THRESHOLD,
        )
    if height_control is None:
        assert len(warning_list) == 0
    else:
        assert len(warning_list) == 1


def test_threshold_stats_img_surface_with_mask(surf_img_1d, surf_mask_1d):
    """Smoke test threshold_stats_img works on surface with a mask."""
    threshold_stats_img(
        surf_img_1d, height_control="bonferroni", mask_img=surf_mask_1d
    )


@pytest.mark.parametrize(
    "threshold, expected_n_unique_values",
    [
        (2.5, 19),
        ([2.5, 3.5], 23),
        ([2.5, 3.0, 3.5], 27),
    ],
)
def test_cluster_level_inference_surface_realistic_data(
    threshold, expected_n_unique_values
):
    """Check cluster_level_inference on realistic data."""
    stat_img = load_fsaverage_data(data_type="thickness")
    th_map = cluster_level_inference(stat_img, threshold=threshold)
    vals = get_surf_data(th_map)
    assert len(np.unique(vals)) == expected_n_unique_values


def test_threshold_stats_img_surface_output(surf_img_1d):
    """Check output threshold_stats_img surface with no height_control.

    Also check the user of cluster_threshold.
    """
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

    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, threshold=2, cluster_threshold=2
    )

    assert_equal(result.data.parts["left"], np.asarray([0.0, 0.0, 3.0, 4.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, 0.0, 6.0, 8.0, 0.0])
    )

    # one sided positive
    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, two_sided=False
    )

    assert_equal(result.data.parts["left"], np.asarray([0.0, 0.0, 0.0, 4.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, 0.0, 6.0, 8.0, 0.0])
    )

    result, _ = threshold_stats_img(
        surf_img_1d,
        height_control=None,
        two_sided=False,
        cluster_threshold=2,
    )

    assert_equal(result.data.parts["left"], np.asarray([0.0, 0.0, 0.0, 0.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, 0.0, 6.0, 8.0, 0.0])
    )

    # one sided negative
    result, _ = threshold_stats_img(
        surf_img_1d, height_control=None, threshold=-0.5, two_sided=False
    )

    assert_equal(result.data.parts["left"], np.asarray([0.0, -1.0, 0.0, 0.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, -2.0, 0.0, 0.0, 0.0])
    )

    result, _ = threshold_stats_img(
        surf_img_1d,
        height_control=None,
        threshold=-0.5,
        two_sided=False,
        cluster_threshold=3,
    )

    assert_equal(result.data.parts["left"], np.asarray([0.0, 0.0, 0.0, 0.0]))
    assert_equal(
        result.data.parts["right"], np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
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

    # TODO (nilearn >= 0.15.0)
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


@pytest.mark.slow
@pytest.mark.parametrize("threshold", [3, 3.0, 2.9, DEFAULT_Z_THRESHOLD])
def test_deprecation_threshold_cluster_level_inference(
    threshold, img_3d_rand_eye, surf_img_1d
):
    """Check cluster_level_inference warns when threshold==old threshold .

    # TODO (nilearn >= 0.15.0)
    # remove
    """
    for stat_img in [img_3d_rand_eye, surf_img_1d]:
        with warnings.catch_warnings(record=True) as warning_list:
            cluster_level_inference(stat_img, threshold=threshold)

        n_warnings = len(
            [x for x in warning_list if issubclass(x.category, FutureWarning)]
        )
        if threshold == 3.0:
            assert n_warnings == 1
        else:
            assert n_warnings == 0
