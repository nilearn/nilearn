"""
Parametric modulation
=====================

More specifically:

1. Download an :term:`fMRI` :term:`BIDS` dataset
   with derivatives from openneuro.
2. Perform a typical GLM.
2. Perform a GLM using reaction time as parametric modulation
"""

from nilearn.datasets import (
    fetch_ds000030_urls,
    fetch_openneuro_dataset,
    select_from_index,
)

_, urls = fetch_ds000030_urls()

exclusion_patterns = [
    "*group*",
    "*phenotype*",
    "*mriqc*",
    "*parameter_plots*",
    "*physio_plots*",
    "*space-fsaverage*",
    "*space-T1w*",
    "*dwi*",
    "*beh*",
    "*task-bart*",
    "*task-rest*",
    "*task-scap*",
    "*task-task*",
]
urls = select_from_index(
    urls, exclusion_filters=exclusion_patterns, n_subjects=1
)

data_dir, _ = fetch_openneuro_dataset(urls=urls)

from nilearn.glm.first_level import first_level_from_bids

task_label = "stopsignal"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep"
(
    models,
    models_run_imgs,
    models_events,
    models_confounds,
) = first_level_from_bids(
    data_dir,
    task_label,
    space_label,
    smoothing_fwhm=6.0,
    derivatives_folder=derivatives_folder,
    n_jobs=1,
    verbose=0,
)

# %%
# Access the model and model arguments of the subject and process events.
import pandas as pd

events: list[pd.DataFrame]
model, imgs, events, confounds = (
    models[0],
    models_run_imgs[0],
    models_events[0],
    models_confounds[0],
)
subject = f"sub-{model.subject_label}"
model.minimize_memory = False  # override default

events[0] = events[0].dropna()

# %%
# First level model estimation
# ------------------------------------------
# We fit the first level model for one subject.
model.fit(imgs, events=events)

# %%
report = model.generate_report(
    contrasts=["GO", "STOP"],
    height_control=None,
    threshold=3,
    cluster_threshold=10,
    plot_type="glass",
    title="reference model",
)
report.open_in_browser()


# %%
# First level model estimation with parametric modulation
# ------------------------------------------
events[0] = events[0].rename(columns={"ReactionTime": "modulation"})
model.fit(imgs, events=events)

report = model.generate_report(
    contrasts=["GO", "STOP"],
    height_control=None,
    threshold=3,
    cluster_threshold=10,
    plot_type="glass",
    title="with parametric modulation",
)
report.open_in_browser()
