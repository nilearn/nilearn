"""
Beta-Series Modeling for Task-Based Functional Connectivity
===========================================================

This example shows how to run :term:`beta`-series :term:`GLM` models,
which are a common modeling approach for a variety of analyses of
task-based :term:`fMRI` data with an event-related task design,
including :term:`functional connectivity`, :term:`decoding <Decoding>`,
and Representational Similarity Analysis (RSA).

First, we compare the standard :term:`GLM` modeling approach
to two common beta-series modeling approaches,
Least Squares All (LSA) and Least Squares Separate (LSS).
Then, we show how to use the resulting beta-series for a simple
task-based functional connectivity analysis.
"""

# %%
# Prepare data and analysis parameters
# ------------------------------------
# Download data in :term:`BIDS` format and event information for one subject,
# and create a standard :class:`~nilearn.glm.first_level.FirstLevelModel`.
#
# For more information
# see the :ref:`dataset description <language_localizer_dataset>`.
#
from nilearn.datasets import fetch_language_localizer_demo_dataset
from nilearn.glm.first_level import FirstLevelModel, first_level_from_bids
from nilearn.plotting import plot_design_matrix, plot_stat_map, show

data = fetch_language_localizer_demo_dataset()

# %%
# Find the first subject's functional run, confounds, and events file
# and use them to instantiate a
# :class:`~nilearn.glm.first_level.FirstLevelModel`.
models, models_run_imgs, events_dfs, models_confounds = first_level_from_bids(
    dataset_path=data.data_dir,
    task_label="languagelocalizer",
    space_label="",
    sub_labels=["01"],
    img_filters=[("desc", "preproc")],
    n_jobs=2,
)

# Grab the first subject's FirstLevelModel object
for _idx, model in enumerate(models):
    standard_glm = model

    # Multiple fMRI runs (and accompanying event files) may
    # support a single subject's FirstLevelModel,
    # so we loop through. Note that for this dataset,
    # we have only one run per subject.
    for run_imgs, events in zip(
        models_run_imgs[_idx], events_dfs[_idx], strict=False
    ):
        fmri_file = run_imgs
        events_df = events

# We can verify that only one fMRI run was found for
# this subject as expected.
print("fmri_file:", fmri_file)

# %%
# Define the standard model
# -------------------------
# Here, we create a basic :term:`GLM` for this one run,
# which we can use to highlight differences between standard
# event-modeling approach and beta-series models.
#
# We will just use the one created by
# :func:`~nilearn.glm.first_level.first_level_from_bids`.
import matplotlib.pyplot as plt

# Fit the model.
standard_glm.fit(fmri_file, events_df)

# The standard design matrix has one column for each condition, along with
# columns for the confound regressors and drifts.
fig, ax = plt.subplots(figsize=(5, 10))
plot_design_matrix(standard_glm.design_matrices_[0], axes=ax)
show()

# %%
# We will reuse the parameters supplied above to
# `first_level_from_bids` for all other models, so
# we can extract these directly from the `standard_glm`
# object.
glm_parameters = standard_glm.get_params()

# We need to override one parameter (``signal_scaling``).
glm_parameters["signal_scaling"] = standard_glm.signal_scaling

# %%
# Define the Least Squares-All (LSA) model
# ----------------------------------------
# We will now create a Least Squares-All (LSA) model.
# This involves creating a new condition in the design matrix
# for each trial of interest.
# It's important to ensure that the original conditions can be inferred from
# the new trial-wise conditions, in order to collect the resulting
# :term:`beta` maps into condition-wise beta-series.
# Here, we will do this using a unique delimiter (``__``),
# which should not be present in the original condition names.

# Transform the data frame for LSA
lsa_events_df = events_df.copy()
conditions = lsa_events_df["trial_type"].unique()
condition_counter = dict.fromkeys(conditions, 0)
for i_trial, trial in lsa_events_df.iterrows():
    trial_condition = trial["trial_type"]
    condition_counter[trial_condition] += 1
    # We use a unique delimiter here (``__``) which shouldn't be in the
    # original condition names.
    trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
    lsa_events_df.loc[i_trial, "trial_type"] = trial_name

lsa_glm = FirstLevelModel(**glm_parameters)
lsa_glm.fit(fmri_file, lsa_events_df)

fig, ax = plt.subplots(figsize=(10, 10))
plot_design_matrix(lsa_glm.design_matrices_[0], axes=ax)
show()

# %%
# Aggregate beta maps from the LSA model based on condition
# `````````````````````````````````````````````````````````
# Collect the :term:`Parameter Estimate` maps.
from nilearn.image import concat_imgs

lsa_beta_maps = {cond: [] for cond in events_df["trial_type"].unique()}
trialwise_conditions = lsa_events_df["trial_type"].unique()
for condition in trialwise_conditions:
    beta_map = lsa_glm.compute_contrast(condition, output_type="effect_size")
    # Drop the trial number from the condition name to get the original name,
    # splitting on our delimiter (``__``).
    condition_name = condition.split("__")[0]
    lsa_beta_maps[condition_name].append(beta_map)

# We concatenate the lists of 3D maps into a single 4D beta-series for
# each condition.
lsa_beta_maps = {
    name: concat_imgs(maps) for name, maps in lsa_beta_maps.items()
}

# %%
# Define the Least Squares-Separate (LSS) models
# ----------------------------------------------
# We next create a separate Least Squares-Separate (LSS) model,
# one for each trial in the conditions of interest.
# This is much like the LSA approach,
# except that we only relabel *one* trial in the `events_df` data frame.
# We loop through the trials, create a version of the data frame where the
# targeted trial has a unique trial type, fit the model to that data frame,
# and finally collect the targeted trial's beta map for the beta-series.


def lss_transformer(events_df, row_number):
    """Label one trial for one LSS model.

    Parameters
    ----------
    df : pandas.DataFrame
        BIDS-compliant events file information.
    row_number : int
        Row number in the data frame.
        This indexes the trial that will be isolated.

    Returns
    -------
    df : pandas.DataFrame
        Update events information,
        with the selected trial's trial type isolated.
    trial_name : :obj:`str`
        Name of the isolated trial's trial type.
    """
    events_df = events_df.copy()

    # Determine which number trial it is *within the condition of interest*.
    trial_condition = events_df.loc[row_number, "trial_type"]
    trial_type_series = events_df["trial_type"]
    trial_type_series = trial_type_series.loc[
        trial_type_series == trial_condition
    ]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)

    # We again use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    # Technically, all you need is for the requested trial to have a unique
    # 'trial_type' *within* the data frame, rather than across models.
    # However, we may want to have meaningful 'trial_type's (e.g., 'Left_001')
    # across models, so that we could track individual trials across models.
    trial_name = f"{trial_condition}__{trial_number:03d}"
    events_df.loc[row_number, "trial_type"] = trial_name
    return events_df, trial_name


# Loop through the trials of interest and transform the data frame for LSS.
lss_beta_maps = {cond: [] for cond in events_df["trial_type"].unique()}
lss_design_matrices = []

for i_trial in range(events_df.shape[0]):
    lss_events_df, trial_condition = lss_transformer(events_df, i_trial)

    # Compute and collect beta maps.
    lss_glm = FirstLevelModel(**glm_parameters)
    lss_glm.fit(fmri_file, lss_events_df)

    # We save the design matrices across trials to visualize them later.
    lss_design_matrices.append(lss_glm.design_matrices_[0])

    beta_map = lss_glm.compute_contrast(
        trial_condition,
        output_type="effect_size",
    )

    # Drop the trial number from the condition name to get the original name,
    # splitting on our delimiter (``__``)
    condition_name = trial_condition.split("__")[0]
    lss_beta_maps[condition_name].append(beta_map)

# We again concatenate the lists of 3D maps into a single 4D beta-series for
# each condition.
lss_beta_maps = {
    name: concat_imgs(maps) for name, maps in lss_beta_maps.items()
}

# %%
# Show the LSS design matrices for the first few trials
# `````````````````````````````````````````````````````
fig, axes = plt.subplots(ncols=3, figsize=(40, 20))
for i_trial in range(3):
    plot_design_matrix(
        lss_design_matrices[i_trial],
        axes=axes[i_trial],
    )
    axes[i_trial].set_title(f"Trial {i_trial + 1}")

show()

# %%
# Compare design matrices from the three modeling approaches
# ----------------------------------------------------------
DM_TITLES = ["Standard GLM", "LSA Model", "LSS Model (Trial 1)"]
DESIGN_MATRICES = [
    standard_glm.design_matrices_[0],
    lsa_glm.design_matrices_[0],
    lss_design_matrices[0],
]

fig, axes = plt.subplots(
    ncols=3,
    figsize=(40, 20),
    gridspec_kw={"width_ratios": [1, 2, 1]},
)

for i_ax, _ in enumerate(axes):
    plot_design_matrix(DESIGN_MATRICES[i_ax], axes=axes[i_ax])
    axes[i_ax].set_title(DM_TITLES[i_ax])

show()

# %%
# Applications of beta-series
# ---------------------------
# Beta-series can be used much like :term:`resting-state` data,
# though generally with vastly reduced degrees of freedom
# compared to a typical :term:`resting-state` run,
# given that the number of trials should always be less
# than the number of volumes in a :term:`fMRI` run.
#
# Two common applications of beta-series are
# to :term:`functional connectivity` and decoding analyses.
# For an example of a beta-series applied to decoding, see
# :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_glm_decoding.py`.
# Here, we show how the beta-series can be applied to functional connectivity
# analysis.
#
# In the following section, we perform a task-based functional
# connectivity analysis using the two task conditions
# ("language" and "string") from the LSS beta-series.
# The seed coordinate is chosen based on a previous
# `Neurosynth <https://neurosynth.org/>`_ meta-analysis.
# This section is based on
# :ref:`sphx_glr_auto_examples_03_connectivity\
# _plot_seed_to_voxel_correlation.py`,
# which goes into more detail about seed-to-voxel functional connectivity
# analyses.
import numpy as np

from nilearn.maskers import NiftiMasker, NiftiSpheresMasker

# Use coordinate taken from Neurosynth's "language" meta-analysis.
coords = [(-54, -42, 3)]

# Initialize masker for the Neurosynth seed.
seed_masker = NiftiSpheresMasker(
    coords,
    radius=8,
    detrend=True,
    standardize="zscore_sample",
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

# Initialize a separate masker for the whole brain.
brain_masker = NiftiMasker(
    smoothing_fwhm=6,
    detrend=True,
    standardize="zscore_sample",
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

# %%
# Perform the seed-to-voxel correlation for the LSS beta-series.
# ``````````````````````````````````````````````````````````````
# Using the defined ``seed_masker``, we extract the signal from our
# `Neurosynth <https://neurosynth.org/>`_ coordinate for the LSS beta-series
# maps defined for each trial, separately for each task type ("language" and
# "string").
# We then extract the whole-brain signal using the ``brain_masker`` from the
# same LSS beta-series maps.
# Finally, we take the dot-product of these two matrices and normalize it by
# the number of samples; i.e., the number of trials.
#
# We perform this analysis for each task condition separately.
#
language_seed_beta_series = seed_masker.fit_transform(
    lss_beta_maps["language"]
)
language_beta_series = brain_masker.fit_transform(lss_beta_maps["language"])
language_corrs = (
    np.dot(
        language_beta_series.T,
        language_seed_beta_series,
    )
    / language_seed_beta_series.shape[0]
)
language_connectivity_img = brain_masker.inverse_transform(language_corrs.T)

# Perform the same seed-to-voxel correlation for the LSS 'string' beta-series
string_seed_beta_series = seed_masker.fit_transform(lss_beta_maps["string"])
string_beta_series = brain_masker.fit_transform(lss_beta_maps["string"])
string_corrs = (
    np.dot(
        string_beta_series.T,
        string_seed_beta_series,
    )
    / string_seed_beta_series.shape[0]
)
string_connectivity_img = brain_masker.inverse_transform(string_corrs.T)

# %%
# Visualize both correlation maps.
# ````````````````````````````````
fig, axes = plt.subplots(figsize=(10, 8), nrows=2)
conn_imgs = [language_connectivity_img, string_connectivity_img]
conn_img_lbls = ["language", "string"]

for img, lbl, ax in zip(conn_imgs, conn_img_lbls, axes, strict=False):
    display = plot_stat_map(
        img,
        threshold=0.5,
        vmax=1,
        cut_coords=coords[0],
        title=lbl,
        figure=fig,
        axes=ax,
    )
    display.add_markers(
        marker_coords=coords,
        marker_color="g",
        marker_size=200,
    )

fig.suptitle(
    "Least Squares-Separate (LSS) Beta-Series Functional Connectivity"
)

show()
