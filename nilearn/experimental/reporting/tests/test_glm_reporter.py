import pytest

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
