import pytest

from nilearn.conftest import _img_3d_ones
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_bland_altman


@pytest.mark.parametrize(
    "masker",
    [
        None,
        _img_3d_ones(),
        NiftiMasker(mask_img=_img_3d_ones()),
        NiftiMasker(mask_img=_img_3d_ones()).fit(),
    ],
)
def test_plot_bland_altman(tmp_path, img_3d_rand_eye, masker):
    plot_bland_altman(
        img_3d_rand_eye,
        img_3d_rand_eye,
        masker=masker,
        ref_label="image set 1",
        src_label="image set 2",
        title="cheese shop",
        gridsize=100,
        output_file=tmp_path / "spam.jpg",
    )
