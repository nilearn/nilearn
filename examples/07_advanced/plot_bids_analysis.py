"""
BIDS dataset first- and second-level analysis
=============================================

This example provides a step-by-step walk-through of fitting
a first- and second-level :term:`GLM` to perform
massively univariate statistical analysis of a :term:`BIDS`
dataset, then visualizing the results.
Full details about the :term:`BIDS` standard can be consulted at
`https://bids.neuroimaging.io/ <https://bids.neuroimaging.io/>`_.

More specifically, this example will be divided into three sections:

1. Downloading an :term:`fMRI` :term:`BIDS` dataset
   in :term:`MNI` space, with two task conditions to contrast.
2. Extracting :term:`GLM` first-level model objects automatically
   from the :term:`BIDS` dataset.
3. Fitting a :term:`GLM` second-level model directly from
   the fitted :term:`GLM` first-level models.
"""

# %%
# Fetch example :term:`BIDS` dataset
# ----------------------------------
# We download a simplified :term:`BIDS` dataset made available for illustrative
# purposes. It contains only the necessary
# information for each subject to run a statistical analysis using Nilearn.
# Each of the raw data folders  contain ``bold.json`` and ``events.tsv`` files,
# indicating :term:`fMRI` metadata and the timing of the task events,
# respectively.
# The derivatives folders include preprocessed :term:`fMRI`
# files ``preproc.nii`` and their accompanying ``confounds.tsv`` files.
#
# For more information on this dataset, see
# the :func:`~nilearn.datasets.fetch_language_localizer_demo_dataset`
# description.
#
from nilearn.datasets import fetch_language_localizer_demo_dataset

data = fetch_language_localizer_demo_dataset()

# %%
# We can verify the location of the dataset on disk.
print(data.data_dir)

# %%
# Automatically extract ``FirstLevelModel`` objects
# -------------------------------------------------
# Since :term:`BIDS` datasets follow a known file structure,
# we can automatically infer the task structure for a given ``task_label``
# using :func:`~nilearn.glm.first_level.first_level_from_bids`.
#
# Specifically, :func:`~nilearn.glm.first_level.first_level_from_bids`
# will extract the :term:`fMRI` images (``models_run_imgs``),
# events (``models_events``),
# and confounder regressors (``model_confounds``)
# for each subject in the dataset.
#
# These extracted data are used to instantiate a
# :class:`~nilearn.glm.first_level.FirstLevelModel`, one for each subject.
# Here, these are the ``models`` objects.
from nilearn.glm.first_level import first_level_from_bids

task_label = "languagelocalizer"
(
    models,
    models_run_imgs,
    models_events,
    models_confounds,
) = first_level_from_bids(
    data.data_dir,
    task_label,
    img_filters=[("desc", "preproc")],
    n_jobs=2,
    space_label="",
    smoothing_fwhm=8,
)

# %%
# Quick sanity check on the extracted data
# .........................................
# It is good practice to verify that the data extracted
# from the :term:`BIDS` dataset is as expected.
# Note that Nilearn does not run an extensive BIDS validation internally.

# %%
# First, we confirm that each ``model_run_imgs`` list corresponds
# to one subject, as expected.
from pathlib import Path

for _subject_idx, subject_runs in enumerate(models_run_imgs[:2]):
    for run in subject_runs:
        print(Path(run).name)

# %%
# Next, we verify the column headers of the first confounds table;
# i.e., for the first subject.
for _subject_idx, subject_confounds in enumerate(models_confounds[:1]):
    for confounds in subject_confounds:
        print(confounds.columns)

# %%
# Finally, we verify the event structure.
# During this acquisition,
# each subject read blocks of sentences and consonant strings.
# These are the two conditions in the "languagelocalizer" task.
# We verify that there are 12 blocks for each condition
# for the first subject.
for _subject_idx, subject_events in enumerate(models_events[:1]):
    for events in subject_events:
        print(events["trial_type"].value_counts())

# %%
# For a single subject, we can visualize their event structure
# using :func:`nilearn.plotting.plot_event`.
from nilearn.plotting import plot_event

plot_event(events)

# %%
# First-level model estimation
# ............................
# Now we simply fit each first-level :term:`GLM` model each subject.
# We can then plot the task-specific :term:`contrast` (``language - string``).
# Notice that we can define a :term:`contrast`
# using the names of the conditions specified in the ``events`` dataframe.
# Sum, subtraction and scalar multiplication are allowed.

# %%
# Set the threshold as the z-variate with an uncorrected p-value of 0.001.
from scipy.stats import norm

p001_unc = norm.isf(0.001)

# %%
# Plot individual contrast maps.
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from nilearn import plotting

ncols = 2
nrows = ceil(len(models) / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12))
axes = np.atleast_2d(axes)

# lists from `first_level_from_bids` are zipped together to iterate over them
model_and_args = zip(
    models, models_run_imgs, models_events, models_confounds, strict=False
)

# for each subject:
for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
    # fit the GLM
    model.fit(imgs, events, confounds)
    # compute the contrast of interest
    zmap = model.compute_contrast("language - string")
    plotting.plot_glass_brain(
        zmap,
        threshold=p001_unc,
        title=f"sub-{model.subject_label}",
        axes=axes[int(midx / ncols), int(midx % ncols)],
        plot_abs=False,
        colorbar=True,
        display_mode="x",
        vmin=-12,
        vmax=12,
    )
fig.suptitle("Subjects's z_map language network (unc. p<0.001)")
plotting.show()

# %%
# Second-level model estimation
# -----------------------------
# Now, we just have to provide the list of
# fitted :class:`~nilearn.glm.first_level.FirstLevelModel` objects
# to the :class:`~nilearn.glm.second_level.SecondLevelModel` object
# for estimation.
# We can do this because all subjects share a similar design matrix
# (i.e., the same variables represented with identical column names).
from nilearn.glm.second_level import SecondLevelModel

second_level_input = models

# %%
# We apply a smoothing of 8mm and parallelize the computation.
second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=2)
second_level_model = second_level_model.fit(second_level_input)

# %%
# Computing contrasts at the second-level is as simple as at the first-level.
# Since we are not providing confounders,
# we are performing a one-sample test
# at the second-level with the images determined by the specified first-level
# contrast.
zmap = second_level_model.compute_contrast(
    first_level_contrast="language - string"
)

# %%
# The second-level :term:`contrast` reveals a left lateralized fronto-temporal
# language network.
plotting.plot_glass_brain(
    zmap,
    threshold=p001_unc,
    title="Group language network (unc. p<0.001)",
    plot_abs=False,
    figure=plt.figure(figsize=(5, 4)),
)
plotting.show()

# %%
# We can generate and save the second-level :term:`GLM` report.
report_slm = second_level_model.generate_report(
    contrasts="intercept",
    first_level_contrast="language - string",
    threshold=p001_unc,
    height_control=None,
    display_mode="x",
)

# %%
# View the second-level :term:`GLM` report.
#
# .. include:: ../../../examples/report_note.rst
#
report_slm
