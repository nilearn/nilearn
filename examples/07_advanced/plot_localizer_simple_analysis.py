"""
Massively univariate analysis of a calculation task from the Localizer dataset
==============================================================================

This example shows how to perform a standard
:term:`ANOVA` with `scikit-learn <https://scikit-learn.org>`_ and Nilearn.
Using :func:`sklearn.feature_selection.f_regression`,
`a massively univariate F-test
<https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_
is performed; we then threshold and plot the resulting
:term:`Bonferroni-corrected <FPR correction>` p-values.

We use the calculation-task :term:`contrast` maps from the
:ref:`Localizer dataset <brainomics_maps>`,
accessed via the
:func:`~nilearn.datasets.fetch_localizer_calculation_task` fetcher.
For a complete picture of this dataset,
please refer to the :ref:`dataset description <brainomics_maps>`.

This fetcher returns a subset of the broader Localizer task;
note that this dataset contains many other contrast maps as
well as external, subject-related or behavioral variates,
which can be accessed with the
:func:`~nilearn.datasets.fetch_localizer_contrasts` fetcher.
Please refer to the
:ref:`sphx_glr_auto_examples_07_advanced_plot_localizer_mass_univariate_methods.py`
example for an illustration of
how to use these external variates in other massively
univariate analyses.
"""

# %%
# Load Localizer "calculation task" contrast maps
# -----------------------------------------------
# First, we fetch calculation task
# :term:`contrast` maps
# from the
# :func:`~nilearn.datasets.fetch_localizer_calculation_task`
# data fetcher for a subset of subjects.
# Here, we only use :term:`contrast` maps from 20 subjects
# in order to speed up computation.
# Paths on disk for all :term:`contrast` maps are accessed
# via the ``cmaps`` attribute.
#
# We also define ``tested_var`` as an array of ones of shape
# (n_subjects, 1).
#
import numpy as np

from nilearn import datasets

n_subjects = 20
localizer_dataset = datasets.fetch_localizer_calculation_task(
    n_subjects=n_subjects
)
cmap_filenames = localizer_dataset.cmaps

tested_var = np.ones(
    n_subjects,
)

# %%
# Extract :term:`voxelwise <voxel>` data
# --------------------------------------
# Next, we use a :func:`~nilearn.maskers.NiftiMasker`
# to extract voxelwise values for the
# calculation task :term:`contrast` maps
# for each subject.
# We also apply a light processing on this data,
# including smoothing with a 5mm :term:`FWHM` kernel.
#
from nilearn.maskers import NiftiMasker

nifti_masker = NiftiMasker(
    smoothing_fwhm=5, memory="nilearn_cache", memory_level=1, verbose=1
)
fmri_masked = nifti_masker.fit_transform(cmap_filenames)

# %%
# ANOVA (parametric F-scores)
# ---------------------------
# We use :func:`sklearn.feature_selection.f_regression` to perform
# a one-sample F-test at every :term:`voxel` and keep only those
# which are significant, as assessed via a simple F-score.
# Assuming that no such effect exists,
# the F-test follows a
# `Fisher distribution <https://en.wikipedia.org/wiki/F-distribution>`_,
# which yields voxelwise p-values that can be used to assert
# significance.
#
from sklearn.feature_selection import f_regression

_, pvals_anova = f_regression(
    fmri_masked,
    tested_var,
    center=False,  # ``center=False`` to not remove intercept.
)

# %%
# We calculate the negative log of the p-values
# for thresholding and visualization.
#
from nilearn.image import get_data

pvals_anova *= fmri_masked.shape[1]
pvals_anova[np.isnan(pvals_anova)] = 1
pvals_anova[pvals_anova > 1] = 1
neg_log_pvals_anova = -np.log10(pvals_anova)
neg_log_pvals_anova_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_anova
)

threshold = -np.log10(0.1)  # 10% corrected
n_detections = (get_data(neg_log_pvals_anova_unmasked) > threshold).sum()

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

from nilearn.plotting import plot_stat_map, show

title = (
    "Negative $\\log_{10}$ p-values"
    "\n(Parametric + Bonferroni correction)"
    f"\n{n_detections} detections"
)

# We plot a single slice to highlight those voxels
# which survive the one-sample F-test.
plotted_slice = 45
fig = plt.figure(figsize=(5, 6), facecolor="w")

# Plot ANOVA p-values
display = plot_stat_map(
    neg_log_pvals_anova_unmasked,
    threshold=threshold,
    display_mode="z",
    cut_coords=[plotted_slice],
    figure=fig,
    cmap="inferno",
    vmin=threshold,
    title=title,
)

show()
