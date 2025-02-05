"""Tests for :func:`nilearn.plotting.plot_img_comparison`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.conftest import _affine_mni, _img_mask_mni, _make_surface_mask
from nilearn.image import iter_img
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.plotting import plot_bland_altman
from nilearn.plotting.img_comparison import plot_img_comparison

# ruff: noqa: ARG001


def test_deprecation_function_moved(matplotlib_pyplot, img_3d_ones_eye):
    from nilearn.plotting.img_plotting import plot_img_comparison

    with pytest.warns(DeprecationWarning, match="moved"):
        plot_img_comparison(
            [img_3d_ones_eye],
            [img_3d_ones_eye],
            NiftiMasker(img_3d_ones_eye).fit(),
        )


def test_plot_img_comparison(matplotlib_pyplot, rng):
    """Tests for plot_img_comparision."""
    fig, axes = plt.subplots(2, 1)
    axes = axes.ravel()
    kwargs = {"shape": (3, 2, 4), "length": 5}

    query_images, mask_img = generate_fake_fmri(random_state=rng, **kwargs)
    # plot_img_comparison doesn't handle 4d images ATM
    query_images = list(iter_img(query_images))

    target_images, _ = generate_fake_fmri(random_state=rng, **kwargs)
    target_images = list(iter_img(target_images))
    target_images[0] = query_images[0]

    masker = NiftiMasker(mask_img).fit()

    correlations = plot_img_comparison(
        target_images, query_images, masker, axes=axes, src_label="query"
    )

    assert len(correlations) == len(query_images)
    assert correlations[0] == pytest.approx(1.0)
    ax_0, ax_1 = axes
    # 5 scatterplots
    assert len(ax_0.collections) == 5
    assert len(
        ax_0.collections[0].get_edgecolors()
        == masker.transform(target_images[0]).ravel().shape[0]
    )
    assert ax_0.get_ylabel() == "query"
    assert ax_0.get_xlabel() == "image set 1"
    # 5 regression lines
    assert len(ax_0.lines) == 5
    assert ax_0.lines[0].get_linestyle() == "--"
    assert ax_1.get_title() == "Histogram of imgs values"
    assert len(ax_1.patches) == 5 * 2 * 128

    correlations_1 = plot_img_comparison(
        target_images, query_images, masker, plot_hist=False
    )

    assert np.allclose(correlations, correlations_1)


def _mask():
    affine = _affine_mni()
    data_positive = np.zeros((7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = 1
    return Nifti1Image(data_positive, affine)


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

    error_msg = "'masker' must be NiftiMasker or Niimg-Like"
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(img_3d_rand_eye, img_3d_rand_eye, masker=1)

    # invalid masker for that image type
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(
            img_3d_rand_eye, img_3d_rand_eye, masker=SurfaceMasker()
        )
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(
            img_3d_rand_eye, img_3d_rand_eye, masker=surf_mask_1d
        )

    error_msg = "'masker' must be SurfaceMasker or SurfaceImage"
    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(surf_img_1d, surf_img_1d, masker=NiftiMasker())

    with pytest.raises(TypeError, match=error_msg):
        plot_bland_altman(surf_img_1d, surf_img_1d, masker=img_3d_ones_eye)

    with pytest.raises(
        TypeError, match="'lims' must be a list or tuple of length == 4"
    ):
        plot_bland_altman(img_3d_rand_eye, img_3d_rand_eye, lims=[-1])

    with pytest.raises(TypeError, match="with all values different from 0."):
        plot_bland_altman(img_3d_rand_eye, img_3d_rand_eye, lims=[0, 1, -2, 0])
