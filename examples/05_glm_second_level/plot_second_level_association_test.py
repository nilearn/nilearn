"""
Example of generic design in second-level models
================================================

This example shows the results obtained in a group analysis using a more
complex contrast than a one- or two-sample t test.
We use the [left button press (auditory cue)] task from the Localizer
dataset and seek association between the contrast values and a variate
that measures the speed of pseudo-word reading. No confounding variate
is included in the model.
"""

# %%
# At first, we need to load the Localizer contrasts.
from nilearn.datasets import fetch_localizer_contrasts

n_samples = 94
localizer_dataset = fetch_localizer_contrasts(
    ["left button press (auditory cue)"],
    n_subjects=n_samples,
)

# %%
# Let's print basic information on the dataset.
print(
    "First contrast nifti image (3D) is located "
    f"at: {localizer_dataset.cmaps[0]}"
)

# %%
# we also need to load the behavioral variable.
tested_var = localizer_dataset.ext_vars["pseudo"]
print(tested_var)

# %%
# It is worth to do a quality check and remove subjects with missing values.
import numpy as np

mask_quality_check = np.where(np.logical_not(np.isnan(tested_var)))[0]
n_samples = mask_quality_check.size
contrast_map_filenames = [
    localizer_dataset.cmaps[i] for i in mask_quality_check
]
tested_var = tested_var[mask_quality_check].to_numpy().reshape((-1, 1))
print(f"Actual number of subjects after quality check: {int(n_samples)}")

# %%
# Estimate second level model
# ---------------------------
# We define the input maps and the design matrix for the second level model
# and fit it.
import pandas as pd

design_matrix = pd.DataFrame(
    np.hstack((tested_var, np.ones_like(tested_var))),
    columns=["fluency", "intercept"],
)

# %%
# Fit of the second-level model
from nilearn.glm.second_level import SecondLevelModel

model = SecondLevelModel(smoothing_fwhm=5.0, n_jobs=2, verbose=1)
model.fit(contrast_map_filenames, design_matrix=design_matrix)

# %%
# To estimate the :term:`contrast` is very simple.
# We can just provide the column name of the design matrix.
z_map = model.compute_contrast("fluency", output_type="z_score")

# %%
# We compute the fdr-corrected p = 0.05 threshold for these data.
from nilearn.glm import threshold_stats_img

_, threshold = threshold_stats_img(z_map, alpha=0.05, height_control="fdr")

# %%
# Let us plot the second level :term:`contrast` at the computed thresholds.
from nilearn.plotting import plot_stat_map, show

cut_coords = [10, -5, 10]

plot_stat_map(
    z_map,
    threshold=threshold,
    title="Group-level association between motor activity \n"
    "and reading fluency (fdr=0.05)",
    cut_coords=cut_coords,
    draw_cross=False,
)

show()

# %%
# Computing the (corrected) p-values with parametric test to compare with
# non parametric test
from nilearn.image import get_data, math_img

p_val = model.compute_contrast("fluency", output_type="p_value")
n_voxels = np.sum(get_data(model.mask_img_))
# Correcting the p-values for multiple testing and taking negative logarithm
neg_log_pval = math_img(
    f"-np.log10(np.minimum(1, img * {n_voxels!s}))", img=p_val
)

# %%
# Let us plot the (corrected) negative log  p-values for the parametric test

# Since we are plotting negative log p-values and using a threshold equal to 1,
# it corresponds to corrected p-values lower than 10%, meaning that there
# is less than 10% probability to make a single false discovery
# (90% chance that we make no false discoveries at all).
# This threshold is much more conservative than the previous one.
threshold = 1
title = (
    "Group-level association between motor activity and reading: \n"
    "neg-log of parametric corrected p-values (FWER < 10%)"
)
plot_stat_map(
    neg_log_pval,
    cut_coords=cut_coords,
    threshold=threshold,
    title=title,
    vmin=threshold,
    cmap="inferno",
    draw_cross=False,
)
show()

# %%
# Computing the (corrected) negative log p-values with permutation test
from nilearn.glm.second_level import non_parametric_inference

neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(
    contrast_map_filenames,
    design_matrix=design_matrix,
    second_level_contrast="fluency",
    model_intercept=True,
    n_perm=1000,
    two_sided_test=False,
    mask=None,
    smoothing_fwhm=5.0,
    n_jobs=2,
    verbose=1,
)

# %%
# Let us plot the (corrected) negative log  p-values
title = (
    "Group-level association between motor activity and reading: \n"
    "neg-log of non-parametric corrected p-values (FWER < 10%)"
)
plot_stat_map(
    neg_log_pvals_permuted_ols_unmasked,
    cut_coords=cut_coords,
    threshold=threshold,
    title=title,
    vmin=threshold,
    cmap="inferno",
    draw_cross=False,
)
show()

# The neg-log p-values obtained with non parametric testing are capped at 3
# since the number of permutations is 1e3.
# The non parametric test yields a few more discoveries
# and is then more powerful than the usual parametric procedure.
