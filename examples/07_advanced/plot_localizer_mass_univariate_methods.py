"""
Massively univariate analysis of a motor task from the Localizer dataset
========================================================================

This example compares results obtained with a massively univariate
analysis (:func:`~nilearn.mass_univariate.permuted_ols`)
after two permutation test correction methods:
Max :term:`t-statistic <Parameter estimate>` Family-wise Error
(:term:`FWE <FWER correction>`) and
Max :term:`TFCE` :term:`FWE <FWER correction>`.

These two methods are compared against a baseline parametric test
(Bonferroni :term:`FWE <FWER correction>`).

The example is structured as follows:

1. First, an
   `ANOVA <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_
   is performed for feature selection and to generate the baseline model.
   Here, the effect of each regressor is evaluated sequentially.

2. Next, we use a permuted Ordinary Least Squares
   (:sklearn:`OLS <modules/linear_model.html#ordinary-least-squares>`)
   analysis,
   run at each :term:`voxel` with
   :func:`~nilearn.mass_univariate.permuted_ols`.
   This model explicitly tests whether or not a voxel responds differently
   under different conditions of a visual task.

We use the ``left button press (auditory cue)`` task contrast maps from the
Localizer dataset (:func:`~nilearn.datasets.fetch_localizer_contrasts`).
This dataset includes external, behavioral variates (``ext_vars``); we
therefore evaluate the association between a behavioral variate that
measures the speed  of pseudo-word reading (``pseudo``) and the
:term:`contrast` map values, at every :term:`voxel`.
"""

# %%
# Load Localizer contrast
# -----------------------
# First, we fetch all ``left button press (auditory cue)``
# contrast maps and associated ``pseudo`` behavioral variates
# from the
# :func:`~nilearn.datasets.fetch_localizer_contrasts`
# data fetcher.
#
import numpy as np

from nilearn import datasets

localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left button press (auditory cue)"]
)

# %%
behav_var = localizer_dataset.ext_vars["pseudo"].values

# Examine the behavioral variate
print(behav_var)

# %%
# Remove subjects without behavioral variate
# ``````````````````````````````````````````
# We see that some subjects do not have scores for this behavioral
# variate. We therefore need to remove them from our analyses.

quality_mask = np.isfinite(behav_var)
n_samples = np.sum(quality_mask)
print(f"Actual number of subjects after quality check: {int(n_samples)}")

# %%
# We cast list of ``cmaps`` to numpy array for Boolean masking with
# ``quality_mask``. Similarly, we subset ``behav_var`` with
# ``quality_mask`` and then reshape from shape (n_samples,) to
# shape (n_samples, 1).
contrast_map_filenames = np.array(localizer_dataset.cmaps)[quality_mask]
tested_var = behav_var[quality_mask].reshape((-1, 1))

# %%
# Extract :term:`voxelwise <voxel>` data
# --------------------------------------
# Next, we use a :func:`~nilearn.maskers.NiftiMasker`
# to extract voxelwise values for the
# ``left button press (auditory cue)`` task :term:`contrast`
# map for each subject who passed quality control.
# We also apply a light processing on this data,
# including smoothing with a 5mm :term:`FWHM` kernel.
#
# Note that we use a :func:`~nilearn.maskers.NiftiMasker` object
# rather than a :func:`~nilearn.maskers.MultiNiftiMasker` object
# in order to extract all :term:`voxel` values for all subjects
# into the same array.
#
from nilearn.maskers import NiftiMasker

nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
fmri_masked = nifti_masker.fit_transform(contrast_map_filenames)


# %%
# :term:`ANOVA` (parametric F-scores)
# -----------------------------------
# We use :func:`sklearn.feature_selection.f_regression` to perform
# :sklearn:`feature selection <modules/feature_selection.html>`
# at every :term:`voxel` and keep only those most related
# to the behavioral variate, as assessed via a simple F-score.
# Assuming that no such effect exists,
# the F-test follows a
# `Fisher distribution <https://en.wikipedia.org/wiki/F-distribution>`_,
# which yields voxelwise p-values that can be used to assert
# significance.
#
from sklearn.feature_selection import f_regression

_, pvals_anova = f_regression(fmri_masked, tested_var.ravel(), center=True)

# %%
# Calculate the negative log of the p-values
# for equivalent visualization with
# :func:`~nilearn.mass_univariate.permuted_ols`.
pvals_anova *= fmri_masked.shape[1]
pvals_anova[np.isnan(pvals_anova)] = 1
pvals_anova[pvals_anova > 1] = 1
neg_log_pvals_anova = -np.log10(pvals_anova)
neg_log_pvals_anova_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_anova
)

# %%
# Perform massively univariate analysis with permuted OLS
# -------------------------------------------------------
# This method will produce both voxel-level Family-wise Error
# (:term:`FWE <FWER correction>`) corrected  negative-log p-values and
# :term:`TFCE`-based :term:`FWE <FWER correction>`-corrected
# negative-log p-values.
#
# .. note::
#   :func:`~nilearn.mass_univariate.permuted_ols` can support a wide range
#   of analysis designs, depending on the numerical labels in ``tested_var``.
#   For example, if you wished to perform a one-sample test, you could
#   simply provide an array of ones (e.g., ``np.ones(n_samples)``).
#
from nilearn.mass_univariate import permuted_ols

ols_outputs = permuted_ols(
    tested_var,  # this is equivalent to the design matrix, in array form
    fmri_masked,
    model_intercept=True,
    masker=nifti_masker,
    tfce=True,
    n_perm=100,  # 100 for the sake of time. Ideally, this should be 10000.
    verbose=1,  # display progress bar
    random_state=0,  # to ensure reproducible results
    n_jobs=2,  # can be changed to use more CPUs
)

# %%
# We select the first regressor from max :term:`t-statistic
# <Parameter estimate>` and assign to the variable
# ``neg_log_pvals_permuted_ols_unmasked`` ; then we perform the same
# procedure for the :term:`TFCE` corrected outputs, assigning to
# the first regressor to ``neg_log_pvals_tfce_unmasked``.
neg_log_pvals_permuted_ols_unmasked = nifti_masker.inverse_transform(
    ols_outputs["logp_max_t"][0, :]
)
neg_log_pvals_tfce_unmasked = nifti_masker.inverse_transform(
    ols_outputs["logp_max_tfce"][0, :]
)

# %%
# Visualization
# -------------
# Since we are plotting negative log p-values and
# using a threshold equal to 1,
# it corresponds to corrected p-values lower than 10%,
# meaning that there is less than 10% probability to
# make a single false discovery
# (i.e., a 90% chance that we make no false discovery at all).
#
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.image import get_data

threshold = -np.log10(0.1)  # 10% corrected

# Calculate the maximum value across all three model variants.
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

fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
for i_col, (title, img) in enumerate(images_to_plot.items()):
    ax = axes[i_col]
    n_detections = (get_data(img) > threshold).sum()
    new_title = f"{title}\n{n_detections} sig. voxels"

    plotting.plot_glass_brain(
        img,
        vmax=vmax,
        display_mode="z",
        threshold=threshold,
        vmin=threshold,
        cmap="inferno",
        figure=fig,
        axes=ax,
    )
    ax.set_title(new_title, pad=10.0)

fig.suptitle(
    "Group left button press ($-\\log_{10}$ p-values)",
    y=1,
    fontsize=16,
)

fig.subplots_adjust(top=0.75, wspace=0.8)

plotting.show()
