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
See :cite:t:`cisler2014comparison` for more information about this,
in the context of functional connectivity analyses.

Two of the most well-known beta series modeling methods are
Least Squares- All (LSA) :cite:p:`rissman2004measuring` and
Least Squares- Separate (LSS)
:cite:p:`mumford2012deconvolving,turner2012spatiotemporal`.
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
from nilearn import image, plotting

##############################################################################
# Prepare data and analysis parameters
# ------------------------------------
# Download BIDS-compliant data and event information for one subject, and
# create a standard :class:`~nilearn.glm.first_level.FirstLevelModel`.
from nilearn.datasets import fetch_language_localizer_demo_dataset

data_dir, _ = fetch_language_localizer_demo_dataset()

from nilearn.glm.first_level import first_level_from_bids

models, models_run_imgs, events_dfs, models_confounds = \
    first_level_from_bids(
        data_dir,
        'languagelocalizer',
        img_filters=[('desc', 'preproc')],
    )

# Grab the first subject's model, functional file, and events DataFrame
standard_glm = models[0]
fmri_file = models_run_imgs[0][0]
events_df = events_dfs[0][0]

# We will use :func:`~nilearn.glm.first_level.first_level_from_bids`'s
# parameters for the other models
glm_parameters = standard_glm.get_params()
# We need to override one parameter (signal_scaling) with the value of
# scaling_axis
glm_parameters['signal_scaling'] = standard_glm.scaling_axis

##############################################################################
# Define the standard model
# -------------------------
# Here, we create a basic GLM for this run, which we can use to highlight
# differences between the standard modeling approach and beta series models.
# We will just use the one created by
# :func:`~nilearn.glm.first_level.first_level_from_bids`.
standard_glm.fit(fmri_file, events_df)

# The standard design matrix has one column for each condition, along with
# columns for the confound regressors and drifts
plotting.plot_design_matrix(standard_glm.design_matrices_[0])

##############################################################################
# Define the LSA model
# --------------------
# We will now create a least squares- all (LSA) model.
# This involves a simple transformation, where each trial of interest receives
# its own unique trial type.
# It's important to ensure that the original trial types can be inferred from
# the updated trial-wise trial types, in order to collect the resulting
# beta maps into condition-wise beta series.

# Transform the DataFrame for LSA
lsa_events_df = events_df.copy()
conditions = lsa_events_df['trial_type'].unique()
condition_counter = {c: 0 for c in conditions}
for i_trial, trial in lsa_events_df.iterrows():
    trial_condition = trial['trial_type']
    condition_counter[trial_condition] += 1
    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names
    trial_name = f'{trial_condition}__{condition_counter[trial_condition]:03d}'
    lsa_events_df.loc[i_trial, 'trial_type'] = trial_name

lsa_glm = FirstLevelModel(**glm_parameters)
lsa_glm.fit(fmri_file, lsa_events_df)

plotting.plot_design_matrix(lsa_glm.design_matrices_[0])

##############################################################################
# Aggregate beta maps from the LSA model based on condition
# `````````````````````````````````````````````````````````
# Collect the parameter estimate maps
lsa_beta_maps = {cond: [] for cond in events_df['trial_type'].unique()}
trialwise_conditions = lsa_events_df['trial_type'].unique()
for condition in trialwise_conditions:
    beta_map = lsa_glm.compute_contrast(condition, output_type='effect_size')
    # Drop the trial number from the condition name to get the original name
    condition_name = condition.split('__')[0]
    lsa_beta_maps[condition_name].append(beta_map)

# We can concatenate the lists of 3D maps into a single 4D beta series for
# each condition, if we want
lsa_beta_maps = {
    name: image.concat_imgs(maps) for name, maps in lsa_beta_maps.items()
}

##############################################################################
# Define the LSS models
# ---------------------
# We will now create a separate LSS model for each trial of interest.
# The transformation is much like the LSA approach, except that we only
# relabel *one* trial in the DataFrame.
# We loop through the trials, create a version of the DataFrame where the
# targeted trial has a unique trial type, fit the model to that DataFrame,
# and finally collect the targeted trial's beta map for the beta series.


def lss_transformer(df, row_number):
    """Label one trial for one LSS model.

    Parameters
    ----------
    df : pandas.DataFrame
        BIDS-compliant events file information.
    row_number : int
        Row number in the DataFrame.
        This indexes the trial that will be isolated.

    Returns
    -------
    df : pandas.DataFrame
        Update events information, with the select trial's trial type isolated.
    trial_name : str
        Name of the isolated trial's trial type.
    """
    df = df.copy()

    # Determine which number trial it is *within the condition*
    trial_condition = df.loc[row_number, 'trial_type']
    trial_type_series = df['trial_type']
    trial_type_series = trial_type_series.loc[
        trial_type_series == trial_condition
    ]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)

    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    # Technically, all you need is for the requested trial to have a unique
    # 'trial_type' *within* the dataframe, rather than across models.
    # However, we may want to have meaningful 'trial_type's (e.g., 'Left_001')
    # across models, so that you could track individual trials across models.
    trial_name = f'{trial_condition}__{trial_number:03d}'
    df.loc[row_number, 'trial_type'] = trial_name
    return df, trial_name


# Loop through the trials of interest and transform the DataFrame for LSS
lss_beta_maps = {cond: [] for cond in events_df['trial_type'].unique()}
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
        output_type='effect_size',
    )

    # Drop the trial number from the condition name to get the original name
    condition_name = trial_condition.split('__')[0]
    lss_beta_maps[condition_name].append(beta_map)

# We can concatenate the lists of 3D maps into a single 4D beta series for
# each condition, if we want
lss_beta_maps = {
    name: image.concat_imgs(maps) for name, maps in lss_beta_maps.items()
}

##############################################################################
# Show the design matrices for the first few trials
# `````````````````````````````````````````````````
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=3, figsize=(20, 10))
for i_trial in range(3):
    plotting.plot_design_matrix(
        lss_design_matrices[i_trial],
        ax=axes[i_trial],
    )
    axes[i_trial].set_title(f'Trial {i_trial + 1}')

fig.show()

##############################################################################
# Compare the three modeling approaches
# -------------------------------------

fig, axes = plt.subplots(
    ncols=3,
    figsize=(20, 10),
    gridspec_kw={'width_ratios': [1, 2, 1]},
)

plotting.plot_design_matrix(
    standard_glm.design_matrices_[0],
    ax=axes[0],
)
axes[0].set_title('Standard GLM')

plotting.plot_design_matrix(
    lsa_glm.design_matrices_[0],
    ax=axes[1],
)
axes[1].set_title('LSA Model')

plotting.plot_design_matrix(
    lss_design_matrices[0],
    ax=axes[2],
)
axes[2].set_title('LSS Model (Trial 1)')

fig.show()
