"""
Massively univariate analysis of a calculation task from the Localizer dataset
==============================================================================

This example shows how to use the Localizer dataset in a basic analysis.
A standard `ANOVA <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_
(massively univariate F-test) is performed and the resulting
Bonferroni-corrected p-values are plotted.

We use the calculation task contrast maps from the
Localizer dataset (:func:`~nilearn.datasets.fetch_localizer_calculation_task`).
This dataset includes many other contrast maps and external, subject-related
or behavioral variates, accessible via
:func:`~nilearn.datasets.fetch_localizer_contrasts`.

Please refer to the
:ref:`plot_localizer_mass_univariate_methods` example.
"""

# %%
import numpy as np

from nilearn import datasets
from nilearn.image import get_data
from nilearn.maskers import NiftiMasker

# %%
# Load Localizer contrast
# -----------------------
n_samples = 20
localizer_dataset = datasets.fetch_localizer_calculation_task(
    n_subjects=n_samples
)
tested_var = np.ones((n_samples, 1))

# %%
# Mask data
# ---------
nifti_masker = NiftiMasker(
    smoothing_fwhm=5, memory="nilearn_cache", memory_level=1, verbose=1
)
cmap_filenames = localizer_dataset.cmaps
fmri_masked = nifti_masker.fit_transform(cmap_filenames)

# %%
# Anova (parametric F-scores)
# ---------------------------
from sklearn.feature_selection import f_regression

_, pvals_anova = f_regression(
    fmri_masked,
    tested_var.ravel(),
    center=False,  # ``center=False`` to not remove intercept.
)
pvals_anova *= fmri_masked.shape[1]
pvals_anova[np.isnan(pvals_anova)] = 1
pvals_anova[pvals_anova > 1] = 1
neg_log_pvals_anova = -np.log10(pvals_anova)
neg_log_pvals_anova_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_anova
)

# %%
# Visualization
# -------------
import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map, show

# Various plotting parameters
plotted_slice = 45
threshold = -np.log10(0.1)  # 10% corrected

masked_pvals = np.ma.masked_less(
    get_data(neg_log_pvals_anova_unmasked), threshold
)

title = (
    "Negative $\\log_{10}$ p-values"
    "\n(Parametric + Bonferroni correction)"
    f"\n{(~masked_pvals.mask).sum()} detections"
)

# Plot Anova p-values
display = plot_stat_map(
    neg_log_pvals_anova_unmasked,
    threshold=threshold,
    display_mode="z",
    cut_coords=[plotted_slice],
    figure=plt.figure(figsize=(5, 6), facecolor="w"),
    cmap="inferno",
    vmin=threshold,
    title=title,
)

show()
