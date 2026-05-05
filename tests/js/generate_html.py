"""Generate HTML figure to run JS tests."""

from pathlib import Path

from nilearn.datasets import (
    load_fsaverage_data,
    load_sample_motor_activation_image,
)
from nilearn.maskers import NiftiMasker
from nilearn.plotting import view_img, view_surf

output_path = Path(__file__).parent

fig = view_img(load_sample_motor_activation_image())
fig.save_as_html(output_path / "view_img.html")


fig = view_surf(surf_map=load_fsaverage_data())
fig.save_as_html(output_path / "view_surf.html")


masker = NiftiMasker()
masker.fit(load_sample_motor_activation_image())
report = masker.generate_report()
report.save_as_html(output_path / "nifti_masker_report.html")
