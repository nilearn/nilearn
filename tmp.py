"""Draft."""

import json
from pathlib import Path

from nilearn import plotting
from nilearn.experimental.plotting import plot_surf_stat_map
from nilearn.experimental.surface import (
    SurfaceImage,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.glm.first_level import FirstLevelModel

data_dir = Path("/home/remi/datalad/datasets.datalad.org")
ds000001 = data_dir / "openneuro" / "ds000001"
ds000001_fmriprep = data_dir / "openneuro-derivatives" / "ds000001-fmriprep"

with open(ds000001 / "task-balloonanalogrisktask_bold.json") as sidecar:
    meta = json.load(sidecar)

task = "balloonanalogrisktask"
subjects = ["01"]
space = "fsaverage5"

meshes = load_fsaverage()
fs_data = load_fsaverage_data()

for sub in subjects:
    run_img = SurfaceImage(
        mesh=meshes["pial"],
        data={
            "left": ds000001_fmriprep
            / f"sub-{sub}"
            / "func"
            / (
                f"sub-{sub}_task-{task}_run-1_hemi-L_space-{space}"
                "_bold.func.gii"
            ),
            "right": ds000001_fmriprep
            / f"sub-{sub}"
            / "func"
            / (
                f"sub-{sub}_task-{task}_run-1_hemi-R_space-{space}"
                "_bold.func.gii"
            ),
        },
    )
    events = (
        ds000001
        / f"sub-{sub}"
        / "func"
        / f"sub-{sub}_task-{task}_run-01_events.tsv"
    )

    flm = FirstLevelModel(
        hrf_model="spm + derivative", t_r=meta["RepetitionTime"]
    )

    flm.fit(run_img, events=events)
    output = flm.compute_contrast("pumps_demean")

    # first_image = SurfaceImage(
    #     mesh=meshes["pial"],
    #     data={
    #         "left": run_img.data.parts["left"][150, ...].astype("float32"),
    #         "right": run_img.data.parts["right"][150, ...].astype("float32"),
    #     },
    # )

    # plot_surf_stat_map(first_image, cmap="inferno")
    plot_surf_stat_map(
        output,
        cmap="seismic",
        threshold=3,
        surf_mesh=meshes["inflated"],
        bg_map=fs_data,
    )

    output.to_filename("foo.gii")


plotting.show()
