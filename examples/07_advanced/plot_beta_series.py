"""
Beta-Series Modeling for Task-Based Functional Connectivity and Decoding
========================================================================
This example shows how to run :term:`GLM`s for beta series models, which are a
common modeling approach for a variety of analyses of task-based fMRI
data with an event-related task design, including functional connectivity,
decoding, and representational similarity analysis.

Beta series models fit trial-wise conditions, which allow users to create
"time series" of these trial-wise maps, which can be substituted for the
typical time series used in resting-state functional connectivity analyses.
Generally, these models are most useful for event-related task designs,
while other modeling approaches (e.g., PPI) tend to perform better in block
designs, depending on the type of analysis.
See [Cisler, Bush, & Steele (2014)]
(https://doi.org/10.1016/j.neuroimage.2013.09.018)
for more information about this, in the context of functional connectivity
analyses.

Two of the most well-known beta series modeling methods are
Least Squares- All (LSA) and Least Squares- Separate (LSS).
In LSA, a single :term:`GLM` is run, in which each trial of each condition of
interest is separated out into its own condition within the design matrix.
In LSS, each trial of each condition of interest has its own :term:`GLM`,
in which the targeted trial receives its own column within the design matrix,
but everything else remains the same as the standard model.
Trials are then looped across, and many GLMs are fitted,
with the parameter estimate map extracted from each GLM to build the LSS beta
series.
"""
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image, masking, plotting

##############################################################################
# Prepare data and analysis parameters
# -------------------------------------
# Prepare the data.
from nilearn.datasets import fetch_language_localizer_demo_dataset

data_dir, _ = fetch_language_localizer_demo_dataset()

from nilearn.glm.first_level import first_level_from_bids

models, models_run_imgs, events_dfs, models_confounds = \
    first_level_from_bids(
        data_dir,
        'languagelocalizer',
        img_filters=[('desc', 'preproc')],
    )

standard_glm = models[0]
fmri_file = models_run_imgs[0][0]
events_df = events_dfs[0][0]
confounds_df = models_confounds[0][0]

mask = masking.compute_epi_mask(fmri_file)

# We will use :func:`~nilearn.glm.first_level.first_level_from_bids`'s
# parameters for the other models.
glm_parameters = standard_glm.get_params()
# We need to override one parameter (signal_scaling) with the value of
# scaling_axis.
glm_parameters["signal_scaling"] = standard_glm.scaling_axis

##############################################################################
# Define the standard model
# -------------------------------------
# Here, we create a basic GLM for this run, which we can use to highlight
# differences between the standard modeling approach and beta series models.
# We will just use the one created by
# :func:`~nilearn.glm.first_level.first_level_from_bids`.
standard_glm.fit(fmri_file, events_df)

# The standard design matrix has one column for each condition, along with
# columns for the confound regressors and drifts.
plotting.plot_design_matrix(standard_glm.design_matrices_[0])

##############################################################################
# Define the LSA model
# -------------------------------------
# We will now create an LSA model.

# Transform the DataFrame for LSA
lsa_events_df = events_df.copy()
conditions = lsa_events_df["trial_type"].unique()
lsa_events_df["old_trial_type"] = lsa_events_df["trial_type"]
condition_counter = {c: 0 for c in conditions}
for i_trial, trial in lsa_events_df.iterrows():
    trial_condition = trial["old_trial_type"]
    condition_counter[trial_condition] += 1
    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    trial_name = f"{trial_condition}__{condition_counter[trial_condition]:03d}"
    lsa_events_df.loc[i_trial, "trial_type"] = trial_name

lsa_glm = FirstLevelModel(**glm_parameters)
lsa_glm.fit(fmri_file, lsa_events_df)

plotting.plot_design_matrix(lsa_glm.design_matrices_[0])

##############################################################################
# Aggregate beta maps from the LSA model based on condition
# `````````````````````````````````````````````````````````
# Collect the parameter estimate maps
lsa_beta_maps = {cond: [] for cond in events_df["trial_type"].unique()}
trialwise_conditions = lsa_events_df["trial_type"].unique()
for condition in trialwise_conditions:
    beta_map = lsa_glm.compute_contrast(condition, output_type="effect_size")
    # Drop the trial number from the condition name to get the original name.
    condition_name = condition.split("__")[0]
    lsa_beta_maps[condition_name].append(beta_map)

# We can concatenate the lists of 3D maps into a single 4D beta series for
# each condition, if we want.
lsa_beta_maps = {
    name: image.concat_imgs(maps) for name, maps in lsa_beta_maps.items()
}

##############################################################################
# Define the LSS models
# -------------------------------------


def lss_transformer(df, row_number):
    """Label one trial for one LSS model."""
    df = df.copy()

    # Technically, all you need is for the requested trial to have a unique
    # "trial_type" *within* the dataframe, rather than across models.
    # However, we may want to have meaningful "trial_type"s (e.g., "Left_001")
    # across models, so that you could track individual trials across models.
    df["old_trial_type"] = df["trial_type"]

    # Determine which number trial it is *within the condition*.
    trial_condition = df.loc[row_number, "old_trial_type"]
    trial_type_series = df["old_trial_type"]
    trial_type_series = trial_type_series.loc[
        trial_type_series == trial_condition
    ]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)
    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    trial_name = f"{trial_condition}__{trial_number:03d}"
    df.loc[row_number, "trial_type"] = trial_name
    return df, trial_name


# Transform the DataFrame for LSS
lss_beta_maps = {cond: [] for cond in events_df["trial_type"].unique()}
lss_design_matrices = []

for i_trial in range(events_df.shape[0]):
    lss_events_df, trial_condition = lss_transformer(events_df, i_trial)

    # Compute and collect beta maps
    lss_glm = FirstLevelModel(**glm_parameters)
    lss_glm.fit(fmri_file, lss_events_df)

    # We will save the design matrices across trials to show them later
    lss_design_matrices.append(lss_glm.design_matrices_[0])

    beta_map = lss_glm.compute_contrast(
        trial_condition,
        output_type="effect_size",
    )

    # Drop the trial number from the condition name to get the original name.
    condition_name = trial_condition.split("__")[0]
    lss_beta_maps[condition_name].append(beta_map)

# We can concatenate the lists of 3D maps into a single 4D beta series for
# each condition, if we want.
lss_beta_maps = {
    name: image.concat_imgs(maps) for name, maps in lss_beta_maps.items()
}

##############################################################################
# Compare the three modeling approaches
# -------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=3, figsize=(20, 10))
for i_trial in range(3):
    plotting.plot_design_matrix(
        lss_design_matrices[0],
        ax=axes[i_trial],
    )
    axes[i_trial].set_title(f"Trial {i_trial + 1}")

fig.show()
