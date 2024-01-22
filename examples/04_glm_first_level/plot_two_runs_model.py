"""
Simple example of two-runs fMRI model fitting
=============================================

Here, we will go through a full step-by-step example of fitting a GLM
to experimental data and visualizing the results.
This is done on two runs of one subject of the FIAC dataset.

For more details on the data,
please see experiment 2 in :footcite:t:`dehaene2006functional`.

Here are the steps we will go through:

1. Set up the GLM
2. Compare run-specific and fixed effects contrasts
3. Compute a range of contrasts across both runs
4. Generate a report

Technically, this example shows how to handle two runs
that contain the same experimental conditions.
The model directly returns a fixed effect
of the statistics across the two runs.
"""

# %%
# Create an output ``results`` in the current working directory.
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_two_runs_model"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")


# %%
# Set up the GLM
# --------------
# Inspecting 'data', we note that there are two runs.
# We will retain those two runs in a list of 4D img objects.
from nilearn.datasets import func

data = func.fetch_fiac_first_level()
fmri_imgs = [data["func1"], data["func2"]]

# %%
# Create a mean image for plotting purpose.
from nilearn.image import mean_img

mean_img_ = mean_img(fmri_imgs[0])

# %%
# The design matrices were pre-computed,
# we simply put them in a list of DataFrames.
import numpy as np
import pandas as pd

design_files = [data["design_matrix1"], data["design_matrix2"]]
design_matrices = [pd.DataFrame(np.load(df)["X"]) for df in design_files]


# %%
# Initialize and run the GLM
# --------------------------
# First, we need to specify the model before fitting it to the data.
# Note that a brain mask was provided in the dataset,
# so that is what we will use.
from nilearn.glm.first_level import FirstLevelModel

fmri_glm = FirstLevelModel(
    mask_img=data["mask"],
    smoothing_fwhm=5,
    minimize_memory=True,
)

# %%
# Compare run-specific and fixed effects contrasts
# ------------------------------------------------
# We can then compare run-specific and fixed effects.
# Here, we compare the activation produced from each run separately
# and then the fixed effects version.
cut_coords = [-129, -126, 49]
contrast_id = "DSt_minus_SSt"

# %%
# Compute the statistics for the first run.
from nilearn import plotting

# Here, we define the contrast of interest for the first run.
# This may differ across runs depending on if the design matrices vary.
contrast_val = [[-1, -1, 1, 1]]

fmri_glm_run_1 = fmri_glm.fit(fmri_imgs[0], design_matrices=design_matrices[0])
summary_statistics_run_1 = fmri_glm_run_1.compute_contrast(
    contrast_val,
    output_type="all",
)
plotting.plot_stat_map(
    summary_statistics_run_1["z_score"],
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f"{contrast_id}, first run",
)

# %%
# Compute the statistics for the second run.
fmri_glm_run_2 = fmri_glm.fit(fmri_imgs[1], design_matrices=design_matrices[1])

contrast_val = np.array([[-1, -1, 1, 1]])

summary_statistics_run_2 = fmri_glm_run_2.compute_contrast(
    contrast_val,
    output_type="all",
)
plotting.plot_stat_map(
    summary_statistics_run_2["z_score"],
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f"{contrast_id}, second run",
)

# %%
# Compute the fixed effects statistics
# using the statistical maps of both runs.
#
# We can use :func:`~nilearn.glm.compute_fixed_effects` to compute
# the fixed effects statistics using the outputs
# from the run-specific FirstLevelModel results.
from nilearn.glm.contrasts import compute_fixed_effects

contrast_imgs = [
    summary_statistics_run_1["effect_size"],
    summary_statistics_run_2["effect_size"],
]
variance_imgs = [
    summary_statistics_run_1["effect_variance"],
    summary_statistics_run_2["effect_variance"],
]

fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(
    contrast_imgs,
    variance_imgs,
    data["mask"],
)
plotting.plot_stat_map(
    fixed_fx_stat,
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f"{contrast_id}, fixed effects",
)

# %%
# Not unexpectedly, the fixed effects version displays higher peaks
# than the input runs.
# Computing fixed effects enhances the signal-to-noise ratio
# of the resulting brain maps.

# %%
# Compute the fixed effects statistics
# using the preprocessed data of both runs.
#
# A more straightforward alternative to fitting run-specific GLMs,
# than combining the results with :func:`~nilearn.glm.compute_fixed_effects`,
# is to simply fit the GLM to both runs at once.
#
# Since we can assume that the design matrices of both runs
# have the same columns, in the same order,
# we can again reuse the first run's contrast vector.
fmri_glm_multirun = fmri_glm.fit(fmri_imgs, design_matrices=design_matrices)

# We can just define the contrast array for one run and assume
# that the design matrix is the same for the other.
# However, if we want to be safe, we should define each contrast separately,
# and provide it as a list.
contrast_val = [
    np.array([[-1, -1, 1, 1]]),  # run 1
    np.array([[-1, -1, 1, 1]]),  # run 2
]

z_map = fmri_glm_multirun.compute_contrast(
    contrast_val,
    output_type="z_score",
)
plotting.plot_stat_map(
    z_map,
    bg_img=mean_img_,
    threshold=3.0,
    cut_coords=cut_coords,
    title=f"{contrast_id}, fixed effects",
)

plotting.show()

# %%
# You may note that the results are the same as the first fixed effects
# analysis, but with a lot less code.

# %%
# Compute a range of contrasts across both runs
# ---------------------------------------------
# It may be useful to investigate a number of contrasts.
# Therefore, we will move beyond the original contrast of interest
# and both define and compute several.

# %%
# Contrast specification
n_columns = design_matrices[0].shape[1]
contrasts = {
    "SStSSp_minus_DStDSp": np.array([[1, 0, 0, -1]]),
    "DStDSp_minus_SStSSp": np.array([[-1, 0, 0, 1]]),
    "DSt_minus_SSt": np.array([[-1, -1, 1, 1]]),
    "DSp_minus_SSp": np.array([[-1, 1, -1, 1]]),
    "DSt_minus_SSt_for_DSp": np.array([[0, -1, 0, 1]]),
    "DSp_minus_SSp_for_DSt": np.array([[0, 0, -1, 1]]),
    "Deactivation": np.array([[-1, -1, -1, -1, 4]]),
    "Effects_of_interest": np.eye(n_columns)[:5, :],  # An F-contrast
}

# %%
# Next, we compute and plot the statistics for these new contrasts.

print("Computing contrasts...")
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print(f"  Contrast {index + 1:02g} out of {len(contrasts)}: {contrast_id}")
    # Estimate the contasts.
    z_map = fmri_glm.compute_contrast(contrast_val, output_type="z_score")

    # Write the resulting stat images to file.
    z_image_path = output_dir / f"{contrast_id}_z_map.nii.gz"
    z_map.to_filename(z_image_path)

# %%
# Generating a report
# -------------------
# Since we have already computed the FirstLevelModel
# and have a number of contrasts,
# we can quickly create a summary report.
report = fmri_glm_multirun.generate_report(
    contrasts,
    bg_img=mean_img_,
    title="two-runs fMRI model fitting",
)

# %%
# We have several ways to access the report:

# report  # This report can be viewed in a notebook
# report.open_in_browser()

# or we can save as an html file
# from pathlib import Path
# output_dir = Path.cwd() / "results" / "plot_oasis"
# output_dir.mkdir(exist_ok=True, parents=True)
# report.save_as_html(output_dir / 'report.html')

# %%
# References
# ----------
#
#  .. footbibliography::
