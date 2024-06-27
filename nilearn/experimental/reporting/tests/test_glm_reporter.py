import pytest

from nilearn._utils.data_gen import basic_paradigm
from nilearn.experimental.reporting.glm_reporter import (
    _make_surface_glm_report,
)
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel


@pytest.mark.parametrize("model", [FirstLevelModel, SecondLevelModel])
def test_empty_reports(tmp_path, model):
    report = _make_surface_glm_report(model())
    report.save_as_html(tmp_path / "tmp.html")
    assert (tmp_path / "tmp.html").exists()


def test_flm_generate_report_error_with_surface_data(
    mini_binary_mask, make_mini_img
):
    """Raise NotImplementedError when generate report is called on surface."""
    model = FirstLevelModel(mask_img=mini_binary_mask, t_r=2.0)
    events = basic_paradigm()

    mini_img = make_mini_img((4,))
    model.fit(mini_img, events=events)

    with pytest.raises(NotImplementedError):
        model.generate_report("c0")

    with pytest.raises(NotImplementedError):
        _make_surface_glm_report(model, "c0")
