import warnings

from nilearn.plotting.html_stat_map import (_view_stat_map_change_warning,
                                            view_stat_map)
from nilearn._utils.testing import generate_fake_fmri


def test_view_stat_map_change_warning():
    with warnings.catch_warnings(record=True) as raised_warnings:
        _view_stat_map_change_warning()
        assert raised_warnings[0].category is FutureWarning


def test_view_stat_map_raises_change_warning():
    fake_4D_mri = generate_fake_fmri(length=1)
    with warnings.catch_warnings(record=True) as raised_warnings:
        view_stat_map(stat_map_img=fake_4D_mri[0])
        assert raised_warnings[0].category is FutureWarning



