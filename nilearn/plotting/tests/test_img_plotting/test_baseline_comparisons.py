"""
Test if public plotting functions' output has changed.

Sometimes, the output of a plotting function may unintentionanly change as a
side effect of changing another function or piece of code that it depends on.
These tests ensure that the outputs are not accidentally changed.

Failures are expected at times when the output is changed intentionally
(e.g. fixing a bug or adding features) for a particular function. In such
cases, the output needs to be manually/visually checked as part of the PR
review process and then a new baseline set for comparison.

Set a new baseline by running:

pytest nilearn/plotting/tests/test_img_plotting/test_baseline_comparisons.py \
--mpl-generate-path=nilearn/plotting/tests/test_img_plotting/baseline

"""

import pytest

from nilearn.datasets import load_sample_motor_activation_image
from nilearn.plotting import (
    plot_anat,
    plot_carpet,
    plot_epi,
    plot_glass_brain,
    plot_img,
    plot_prob_atlas,
    plot_roi,
    plot_stat_map,
)
from nilearn.plotting.img_plotting import MNI152TEMPLATE

PLOTTING_FUNCS_3D = {
    plot_img,
    plot_anat,
    plot_stat_map,
    plot_roi,
    plot_epi,
    plot_glass_brain,
}

PLOTTING_FUNCS_4D = {plot_prob_atlas, plot_carpet}


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "plot_func",
    {
        plot_img,
        plot_stat_map,
        plot_glass_brain,
    },
)
def test_plot_functions_stat_map(plot_func):
    """Smoke tests for 3D plotting functions with default parameters."""
    return plot_func(load_sample_motor_activation_image())


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("plot_func", PLOTTING_FUNCS_3D)
def test_plot_functions_3d_default_params(plot_func, img_3d_mni):
    """Smoke tests for 3D plotting functions with default parameters."""
    return plot_func(img_3d_mni)


@pytest.mark.mpl_image_compare
def test_plot_carpet_default_params(img_4d_mni, img_3d_ones_mni):
    """Smoke-test for 4D plot_carpet with default arguments."""
    return plot_carpet(img_4d_mni, mask_img=img_3d_ones_mni)


@pytest.mark.mpl_image_compare
def test_plot_prob_atlas_default_params(img_3d_mni, img_4d_mni):
    """Smoke-test for plot_prob_atlas with default arguments."""
    return plot_prob_atlas(img_4d_mni, bg_img=img_3d_mni)


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
@pytest.mark.parametrize("display_mode", ["z", "ortho"])
def test_plot_anat_mni(anat_img, display_mode):
    """Tests for plot_anat with MNI template."""
    return plot_anat(anat_img=anat_img, display_mode=display_mode)
