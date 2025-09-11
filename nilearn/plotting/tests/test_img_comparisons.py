"""Tests for nilearn.plotting.img_comparison."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.conftest import _affine_mni, _img_mask_mni, _make_surface_mask
from nilearn.image import iter_img
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.plotting import plot_bland_altman, plot_img_comparison

# ruff: noqa: ARG001


def _mask():
    affine = _affine_mni()
    data_positive = np.zeros((7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = 1
    return Nifti1Image(data_positive, affine)


def test_deprecation_function_moved(matplotlib_pyplot, img_3d_mni):
    from nilearn.plotting.image.img_plotting import (
        plot_img_comparison as old_fn,
    )

    with pytest.warns(DeprecationWarning, match="moved"):
        old_fn(
            img_3d_mni,
            img_3d_mni,
            plot_hist=False,
        )


@pytest.mark.parametrize(
    "masker",
    [
        None,
        _mask(),
        NiftiMasker(mask_img=_img_mask_mni()),
        NiftiMasker(mask_img=_img_mask_mni()).fit(),
    ],
)
def test_plot_img_comparison_masker(matplotlib_pyplot, img_3d_mni, masker):
    """Tests for plot_img_comparison with masker or mask image."""
    plot_img_comparison(
        img_3d_mni,
        img_3d_mni,
        masker=masker,
        plot_hist=False,
    )


@pytest.mark.parametrize(
    "masker",
    [
        None,
        _make_surface_mask(),
        SurfaceMasker(mask_img=_make_surface_mask()),
        SurfaceMasker(mask_img=_make_surface_mask()).fit(),
    ],
)
def test_plot_img_comparison_surface(matplotlib_pyplot, surf_img_1d, masker):
    """Test plot_img_comparison with 2 surface images."""
    plot_img_comparison(
        surf_img_1d, [surf_img_1d, surf_img_1d], masker=masker, plot_hist=False
    )


def test_plot_img_comparison_error(surf_img_1d, img_3d_mni):
    """Err if something else than image or list of image is passed."""
    with pytest.raises(TypeError, match="must both be list of 3D"):
        plot_img_comparison(surf_img_1d, {surf_img_1d})

    with pytest.raises(TypeError, match="must both be list of only"):
        plot_img_comparison(surf_img_1d, img_3d_mni)


@pytest.mark.timeout(0)
def test_plot_img_comparison(matplotlib_pyplot, rng, tmp_path):
    """Tests for plot_img_comparison."""
    _, axes = plt.subplots(2, 1)
    axes = axes.ravel()

    length = 2

    query_images, mask_img = generate_fake_fmri(
        random_state=rng, shape=(2, 3, 4), length=length
    )
    # plot_img_comparison doesn't handle 4d images ATM
    query_images = list(iter_img(query_images))

    target_images, _ = generate_fake_fmri(
        random_state=rng, shape=(4, 5, 6), length=length
    )
    target_images = list(iter_img(target_images))
    target_images[0] = query_images[0]

    masker = NiftiMasker(mask_img).fit()

    correlations = plot_img_comparison(
        target_images,
        query_images,
        masker,
        axes=axes,
        src_label="query",
        output_dir=tmp_path,
        colorbar=False,
    )

    assert len(correlations) == len(query_images)
    assert correlations[0] == pytest.approx(1.0)

    # 5 scatterplots
    ax_0, ax_1 = axes
    assert len(ax_0.collections) == length
    assert len(
        ax_0.collections[0].get_edgecolors()
        == masker.transform(target_images[0]).ravel().shape[0]
    )
    assert ax_0.get_ylabel() == "query"
    assert ax_0.get_xlabel() == "image set 1"

    # 5 regression lines
    assert len(ax_0.lines) == length
    assert ax_0.lines[0].get_linestyle() == "--"
    assert ax_1.get_title() == "Histogram of imgs values"
    gridsize = 100
    assert len(ax_1.patches) == length * 2 * gridsize


@pytest.mark.timeout(0)
def test_plot_img_comparison_without_plot(matplotlib_pyplot, rng):
    """Tests for plot_img_comparison no plot should return same result."""
    _, axes = plt.subplots(2, 1)
    axes = axes.ravel()

    query_images, mask_img = generate_fake_fmri(
        random_state=rng, shape=(2, 3, 4), length=2
    )
    # plot_img_comparison doesn't handle 4d images ATM
    query_images = list(iter_img(query_images))

    target_images, _ = generate_fake_fmri(
        random_state=rng, shape=(2, 3, 4), length=2
    )
    target_images = list(iter_img(target_images))
    target_images[0] = query_images[0]

    masker = NiftiMasker(mask_img).fit()

    correlations = plot_img_comparison(
        target_images, query_images, masker, plot_hist=True, colorbar=False
    )

    correlations_1 = plot_img_comparison(
        target_images, query_images, masker, plot_hist=False
    )

    assert np.allclose(correlations, correlations_1)


@pytest.mark.parametrize(
    "masker",
    [
        None,
        _mask(),
        NiftiMasker(mask_img=_img_mask_mni()),
        NiftiMasker(mask_img=_img_mask_mni()).fit(),
    ],
)
def test_plot_bland_altman(
    matplotlib_pyplot, tmp_path, img_3d_mni, img_3d_mni_as_file, masker
):
    """Test Bland-Altman plot with different masker values.

    Also check non default values for
    labels,
    title
    grid size,
    and output_file.

    Also checks that input images can be nifti image or path.
    """
    plot_bland_altman(
        img_3d_mni,
        img_3d_mni_as_file,
        masker=masker,
        ref_label="image set 1",
        src_label="image set 2",
        title="cheese shop",
        gridsize=10,
        output_file=tmp_path / "spam.jpg",
        lims=[-1, 5, -2, 3],
        colorbar=False,
    )

    assert (tmp_path / "spam.jpg").is_file()


@pytest.mark.parametrize(
    "masker",
    [
        None,
        _make_surface_mask(),
        SurfaceMasker(mask_img=_make_surface_mask()),
        SurfaceMasker(mask_img=_make_surface_mask()).fit(),
    ],
)
def test_plot_bland_altman_surface(matplotlib_pyplot, surf_img_1d, masker):
    """Test Bland-Altman plot with 2 surface images.

    Also checks tuple value for gridsize.
    """
    plot_bland_altman(
        surf_img_1d, surf_img_1d, masker=masker, gridsize=(10, 80)
    )


def test_plot_bland_altman_errors(
    surf_img_1d, surf_mask_1d, img_3d_rand_eye, img_3d_ones_eye
):
    """Check common errors for bland altman plots.

    - both inputs must be niimg like or surface
    - valid masker type for volume or surface data
    """
    error_msg = "'ref_img' and 'src_img' must both be"
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(1, "foo")

    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(surf_img_1d, img_3d_rand_eye)

    with pytest.raises(TypeError, match="Mask should be of type:"):
        plot_bland_altman(img_3d_rand_eye, img_3d_rand_eye, masker=1)

    with pytest.raises(
        TypeError, match="'lims' must be a list or tuple of length == 4"
    ):
        plot_bland_altman(img_3d_rand_eye, img_3d_rand_eye, lims=[-1])

    with pytest.raises(TypeError, match="with all values different from 0."):
        plot_bland_altman(img_3d_rand_eye, img_3d_rand_eye, lims=[0, 1, -2, 0])


def test_plot_bland_altman_incompatible_errors(
    surf_img_1d, surf_mask_1d, img_3d_rand_eye, img_3d_ones_eye
):
    """Check error for bland altman plots incompatible mask and images."""
    error_msg = "Mask and input images must be of compatible types."
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(
            img_3d_rand_eye, img_3d_rand_eye, masker=SurfaceMasker()
        )
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(
            img_3d_rand_eye, img_3d_rand_eye, masker=surf_mask_1d
        )
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(surf_img_1d, surf_img_1d, masker=NiftiMasker())

    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(surf_img_1d, surf_img_1d, masker=img_3d_ones_eye)
