"""
Simple example of two-session fMRI model fitting
================================================

Here, we will go through a full step-by-step example of fitting a GLM to
experimental data and visualizing the results.
This is done on two runs of one subject of the FIAC dataset.

For details on the data, please see :footcite:p:`dehaene2006functional`.

More specifically:

1. A sequence of fMRI volumes is loaded.
2. A design matrix describing all the effects related to the data is computed.
3. A mask of the useful brain volume is computed.
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation).

Technically, this example shows how to handle two sessions that contain the
same experimental conditions. The model directly returns a fixed effect of the
statistics across the two sessions.
"""

###############################################################################
# Create a write directory to work,
# it will be a 'results' subdirectory of the current directory.
from os import mkdir, path, getcwd

write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

###############################################################################
# Prepare data and analysis parameters
# ------------------------------------
# Inspecting 'data', we note that there are two sessions.
# We will retain those two sessions in a list of 4D img objects.
from nilearn.datasets import func

data = func.fetch_fiac_first_level()
fmri_imgs = [data['func1'], data['func2']]

###############################################################################
# Create a mean image for plotting purpose.
from nilearn.image import mean_img

mean_img_ = mean_img(fmri_imgs[0])

###############################################################################
# The design matrices were pre-computed, we simply put them in a list of
# DataFrames.
import numpy as np
import pandas as pd

design_files = [data['design_matrix1'], data['design_matrix2']]
design_matrices = [pd.DataFrame(np.load(df)['X']) for df in design_files]

###############################################################################
# To define the contrasts, we will first use a small function to ease the
# contrast definition.


def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors."""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


###############################################################################
# Initialize the GLM
# ------------------
# First, we need to specify the model before fitting it to the data.
# Note that a brain mask was provided in the dataset, so that is what we will
# use.
from nilearn.glm.first_level import FirstLevelModel

fmri_glm = FirstLevelModel(
    mask_img=data['mask'],
    smoothing_fwhm=5,
    minimize_memory=True,
)

###############################################################################
# We can then compare session-specific and fixed effects.
# Here, we compare the activation mas produced from each session separately and
# then the fixed effects version.
cut_coords = [-129, -126, 49]
contrast_id = 'DSt_minus_SSt'

###############################################################################
# Compute the statistics for the first session.
from nilearn import plotting

# Here, we define the contrast of interest for the first session.
# This may differ across sessions depending on if the design matrices vary.
contrast_val = pad_vector([-1, -1, 1, 1], design_matrices[0].shape[1])

fmri_glm_ses1 = fmri_glm.fit(fmri_imgs[0], design_matrices=design_matrices[0])
summary_statistics_ses1 = fmri_glm_ses1.compute_contrast(
    contrast_val,
    output_type='all',
)
plotting.plot_stat_map(
    summary_statistics_ses1['z_score'],
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f'{contrast_id}, first session',
)

###############################################################################
# Compute the statistics for the second session.
fmri_glm_ses2 = fmri_glm.fit(fmri_imgs[1], design_matrices=design_matrices[1])

contrast_val = pad_vector([-1, -1, 1, 1], design_matrices[1].shape[1])

summary_statistics_ses2 = fmri_glm_ses2.compute_contrast(
    contrast_val,
    output_type='all',
)
plotting.plot_stat_map(
    summary_statistics_ses2['z_score'],
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f'{contrast_id}, second session',
)

###############################################################################
# Compute the fixed effects statistics using both sessions' statistical maps.
#
# We can use :func:`~nilearn.glm.contrasts.compute_fixed_effects` to compute
# the fixed effects statistics using the outputs from the session-specific
# FirstLevelModel results.
from nilearn.glm.contrasts import compute_fixed_effects

contrast_imgs = [
    summary_statistics_ses1['effect_size'],
    summary_statistics_ses2['effect_size'],
]
variance_imgs = [
    summary_statistics_ses1['effect_variance'],
    summary_statistics_ses2['effect_variance'],
]

fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(
    contrast_imgs,
    variance_imgs,
    data['mask'],
)
plotting.plot_stat_map(
    fixed_fx_stat,
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f'{contrast_id}, fixed effects',
)

###############################################################################
# Not unexpectedly, the fixed effects version displays higher peaks than the
# input sessions. Computing fixed effects enhances the signal-to-noise ratio of
# the resulting brain maps.

###############################################################################
# Compute the fixed effects statistics using both sessions' preprocessed data.
#
# A more straightforward alternative to fitting session-specific GLMs, then
# combining the results with
# :func:`~nilearn.glm.contrasts.compute_fixed_effects`, is to simply fit the
# GLM to both sessions at once.
#
# Since we can assume that the design matrices of both sessions have the same
# columns, in the same order, we can again re-use the first session's contrast
# vector.
fmri_glm_multises = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)

# We can just define the contrast array for one session and assume that the
# design matrix is the same for the other.
# However, if we want to be safe, we should define each contrast separately,
# and provide it as a list.
contrast_val = [
    pad_vector([-1, -1, 1, 1], design_matrices[0].shape[1]),  # session 1
    pad_vector([-1, -1, 1, 1], design_matrices[1].shape[1]),  # session 2
]

z_map = fmri_glm_multises.compute_contrast(
    contrast_val,
    output_type='z_score',
)
plotting.plot_stat_map(
    z_map,
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f'{contrast_id}, fixed effects',
)

plotting.show()

###############################################################################
# You may note that the results are the same as the first fixed effects
# analysis, but with a lot less code.

###############################################################################
# Compute a range of contrasts across both sessions
# -------------------------------------------------
# It may be useful to investigate a number of contrasts.
# Therefore, we will move beyond the original contrast of interest and both
# define and compute several.

###############################################################################
# Contrast specification
n_columns = design_matrices[0].shape[1]
contrasts = {
    'SStSSp_minus_DStDSp': pad_vector([1, 0, 0, -1], n_columns),
    'DStDSp_minus_SStSSp': pad_vector([-1, 0, 0, 1], n_columns),
    'DSt_minus_SSt': pad_vector([-1, -1, 1, 1], n_columns),
    'DSp_minus_SSp': pad_vector([-1, 1, -1, 1], n_columns),
    'DSt_minus_SSt_for_DSp': pad_vector([0, -1, 0, 1], n_columns),
    'DSp_minus_SSp_for_DSt': pad_vector([0, 0, -1, 1], n_columns),
    'Deactivation': pad_vector([-1, -1, -1, -1, 4], n_columns),
    'Effects_of_interest': np.eye(n_columns)[:5, :],  # An F-contrast
}

###############################################################################
# Next, we compute and plot the statistics for these new contrasts.

print('Computing contrasts...')
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print(f'  Contrast {index + 1:02g} out of {len(contrasts)}: {contrast_id}')
    # Estimate the contasts.
    z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')

    # Write the resulting stat images to file.
    z_image_path = path.join(write_dir, f'{contrast_id}_z_map.nii.gz')
    z_map.to_filename(z_image_path)

###############################################################################
# Generating a report
# -------------------
# Since we have already computed the FirstLevelModel and have a number of
# contrasts, we can quickly create a summary report.
from nilearn.reporting import make_glm_report

report = make_glm_report(fmri_glm_multises, contrasts, bg_img=mean_img_)

###############################################################################
# We have several ways to access the report:

# report  # This report can be viewed in a notebook
# report.save_as_html('report.html')
# report.open_in_browser()

###############################################################################
# References
# ----------
#
#  .. footbibliography::
