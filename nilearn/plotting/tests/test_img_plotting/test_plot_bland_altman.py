import pytest

from nilearn.conftest import _img_mask_mni, _make_surface_mask
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.plotting import plot_bland_altman

# ruff: noqa: ARG001


@pytest.mark.parametrize(
    "masker",
    [
        None,
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
    # Invalid input
    error_msg = "'ref_img' and 'src_img' must both be"
    with pytest.raises(
        TypeError,
        match=error_msg,
    ):
        plot_bland_altman(
            1,
            "foo",
        )

    # images must be both volumes or surface
    with pytest.raises(
        TypeError,
        match=error_msg,
    ):
        plot_bland_altman(surf_img_1d, img_3d_rand_eye)

    # invalid masker
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
