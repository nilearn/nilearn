"""foo."""

from pathlib import Path

from nilearn.experimental.reporting.glm_reporter import (
    _make_surface_glm_report,
)
from nilearn.glm.first_level import FirstLevelModel

model = FirstLevelModel()

report = _make_surface_glm_report(model)

report.save_as_html(Path() / "results" / "tmp.html")

print(report.body)
