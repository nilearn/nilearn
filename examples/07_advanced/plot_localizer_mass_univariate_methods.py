"""
Massively univariate analysis of a motor task from the Localizer dataset
========================================================================

This example shows the results obtained in a massively univariate
analysis performed at the inter-subject level with various methods.
We use the [left button press (auditory cue)] task from the Localizer
dataset and seek association between the contrast values and a variate
that measures the speed of pseudo-word reading. No confounding variate
is included in the model.

1. A standard :term:`ANOVA` is performed. Data smoothed at 5
   :term:`voxels<voxel>` :term:`FWHM` are used.

2. A permuted Ordinary Least Squares algorithm is run at each :term:`voxel`.
   Data smoothed at 5 :term:`voxels<voxel>` :term:`FWHM` are used.

.. include:: ../../../examples/masker_note.rst

"""

# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, May. 2014
import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols

##############################################################################
# Load Localizer contrast
n_samples = 94
localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left button press (auditory cue)"],
    n_subjects=n_samples,
    legacy_format=False,
)

# print basic information on the dataset
print(
    "First contrast nifti image (3D) is located "
    f"at: {localizer_dataset.cmaps[0]}"
)

tested_var = localizer_dataset.ext_vars["pseudo"]

# Quality check / Remove subjects with bad tested variate
mask_quality_check = np.where(np.logical_not(np.isnan(tested_var)))[0]
n_samples = mask_quality_check.size
contrast_map_filenames = [
    localizer_dataset.cmaps[i] for i in mask_quality_check
]
tested_var = tested_var[mask_quality_check].values.reshape((-1, 1))
print(f"Actual number of subjects after quality check: {int(n_samples)}")

##############################################################################
# Mask data
nifti_masker = NiftiMasker(
    smoothing_fwhm=5, memory="nilearn_cache", memory_level=1
)
fmri_masked = nifti_masker.fit_transform(contrast_map_filenames)


##############################################################################
# Anova (parametric F-scores)
from sklearn.feature_selection import f_regression

_, pvals_anova = f_regression(fmri_masked, tested_var, center=True)
pvals_anova *= fmri_masked.shape[1]
pvals_anova[np.isnan(pvals_anova)] = 1
pvals_anova[pvals_anova > 1] = 1
neg_log_pvals_anova = -np.log10(pvals_anova)
neg_log_pvals_anova_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_anova
)

##############################################################################
# Perform massively univariate analysis with permuted OLS
#
# This method will produce both voxel-level FWE-corrected -log10 p-values and
# :term:`TFCE`-based FWE-corrected -log10 p-values.
#
# .. note::
#   :func:`~nilearn.mass_univariate.permuted_ols` can support a wide range
#   of analysis designs, depending on the ``tested_var``.
#   For example, if you wished to perform a one-sample test, you could
#   simply provide an array of ones (e.g., ``np.ones(n_samples)``).

ols_outputs = permuted_ols(
    tested_var,  # this is equivalent to the design matrix, in array form
    fmri_masked,
    model_intercept=True,
    masker=nifti_masker,
    tfce=True,
    n_perm=200,  # 200 for the sake of time. Ideally, this should be 10000.
    verbose=1,  # display progress bar
    n_jobs=1,  # can be changed to use more CPUs
    output_type="dict",
)
neg_log_pvals_permuted_ols_unmasked = nifti_masker.inverse_transform(
    ols_outputs["logp_max_t"][0, :]  # select first regressor
)
neg_log_pvals_tfce_unmasked = nifti_masker.inverse_transform(
    ols_outputs["logp_max_tfce"][0, :]  # select first regressor
)

##############################################################################
# Visualization
from nilearn import plotting
from nilearn.image import get_data

# Various plotting parameters
z_slice = 12  # plotted slice
threshold = -np.log10(0.1)  # 10% corrected

vmax = max(
    np.amax(ols_outputs["logp_max_t"]),
    np.amax(neg_log_pvals_anova),
    np.amax(ols_outputs["logp_max_tfce"]),
)

images_to_plot = {
    "Parametric Test\n(Bonferroni FWE)": neg_log_pvals_anova_unmasked,
    "Permutation Test\n(Max t-statistic FWE)": (
        neg_log_pvals_permuted_ols_unmasked
    ),
    "Permutation Test\n(Max TFCE FWE)": neg_log_pvals_tfce_unmasked,
}

fig, axes = plt.subplots(figsize=(12, 3), ncols=3)
for i_col, (title, img) in enumerate(images_to_plot.items()):
    ax = axes[i_col]
    n_detections = (get_data(img) > threshold).sum()
    new_title = f"{title}\n{n_detections} sig. voxels"

    plotting.plot_glass_brain(
        img,
        colorbar=True,
        vmax=vmax,
        display_mode="z",
        plot_abs=False,
        cut_coords=[12],
        threshold=threshold,
        figure=fig,
        axes=ax,
    )
    ax.set_title(new_title)

fig.suptitle(
    "Group left button press ($-\\log_{10}$ p-values)",
    y=1.3,
    fontsize=16,
)
