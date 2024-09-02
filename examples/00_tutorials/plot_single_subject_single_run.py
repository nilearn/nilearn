"""
Intro to GLM Analysis: a single-run, single-subject fMRI dataset
================================================================

In this tutorial, we use a General Linear Model (:term:`GLM`) to compare the
:term:`fMRI` signal during periods of auditory stimulation
versus periods of rest.

warning::

    The analyse described here is performed in the native space,
    directly on the original :term:`EPI` scans
    without any spatial or temporal preprocessing.
    More sensitive results would likely be obtained on the corrected,
    spatially normalized and smoothed images.
"""

# %%
# Retrieving the data
# -------------------
#
# .. note:: In this tutorial, we load the data using a data downloading
#           function. To input your own data, you will need to provide
#           a list of paths to your own files in the ``subject_data`` variable.
#           These should abide to the Brain Imaging Data Structure
#           (:term:`BIDS`) organization.

from nilearn.datasets import fetch_spm_auditory

subject_data = fetch_spm_auditory()

# print dataset descriptions
print(subject_data.description)

# print paths of func image
print(subject_data.func[0])

from nilearn.image import mean_img

# %%
# We can display the mean functional image and the subject's anatomy:
from nilearn.plotting import plot_anat, plot_img, plot_stat_map

fmri_img = subject_data.func
mean_img = mean_img(subject_data.func[0], copy_header=True)
plot_img(mean_img, colorbar=True, cbar_tick_format="%i")
plot_anat(subject_data.anat[0], colorbar=True, cbar_tick_format="%i")


# %%
# Specifying the experimental paradigm
# ------------------------------------
#
# We must now provide a description of the experiment, that is,
# define the timing of the auditory stimulation and rest periods.
# This is typically provided in an events.tsv file.
# The path of this file is provided in the dataset.
import pandas as pd

events = pd.read_csv(subject_data["events"][0], sep="\t")
events

# %%
# Performing the :term:`GLM` analysis
# -----------------------------------
#
# It is now time to create and estimate a ``FirstLevelModel`` object,
# that will generate the *design matrix*
# using the information provided by the ``events`` object.

from nilearn.glm.first_level import FirstLevelModel

# %%
# Parameters of the first-level model
#
# * t_r=7(s) is the time of repetition of acquisitions
# * noise_model='ar1' specifies the noise covariance model: a lag-1 dependence
# * standardize=False means that we do not want
#   to rescale the time series to mean 0, variance 1
# * hrf_model='spm' means that we rely
#   on the :term:`SPM` "canonical hrf" model
#   (without time or dispersion derivatives)
# * drift_model='cosine' means that we model the signal drifts
#   as slow oscillating time functions
# * high_pass=0.01(Hz) defines the cutoff frequency
#   (inverse of the time period).
fmri_glm = FirstLevelModel(
    t_r=7,
    noise_model="ar1",
    standardize=False,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=0.01,
)

# %%
# Now that we have specified the model, we can run it on the :term:`fMRI` image
fmri_glm = fmri_glm.fit(fmri_img, events)

# %%
# One can inspect the design matrix (rows represent time, and
# columns contain the predictors).
design_matrix = fmri_glm.design_matrices_[0]

# %%
# Formally, we have taken the first design matrix, because the model is
# implictily meant to for multiple runs.
import matplotlib.pyplot as plt

from nilearn.plotting import plot_design_matrix

plot_design_matrix(design_matrix)

plt.show()

# %%
# Save the design matrix image to disk
# first create a directory where you want to write the images
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_single_subject_single_run"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")

plot_design_matrix(design_matrix, output_file=output_dir / "design_matrix.png")

# %%
# The first column contains the expected response profile of regions which are
# sensitive to the auditory stimulation.
# Let's plot this first column

plt.plot(design_matrix["listening"])
plt.xlabel("scan")
plt.title("Expected Auditory Response")
plt.show()

# %%
# Detecting voxels with significant effects
# -----------------------------------------
#
# To access the estimated coefficients (Betas of the :term:`GLM` model),
# we created :term:`contrast` with a single '1' in each of the columns:
# The role of the :term:`contrast` is to select some columns of the model
# --and potentially weight them-- to study the associated statistics.
# So in a nutshell, a :term:`contrast`
# is a weighted combination of the estimated effects.
# Here we can define canonical contrasts that just consider
# the effect of the stimulation in isolation.
#
# .. note::
#
#       Here the baseline is implicit, so passing a value of 1
#       for the first column will give contrast for: ``listening > rest``
#

import numpy as np

nb_regressors = design_matrix.shape[1]
activation = np.zeros(nb_regressors)
activation[0] = 1

# %%
# Let's look at it: plot the coefficients of the :term:`contrast`, indexed by
# the names of the columns of the design matrix.

from nilearn.plotting import plot_contrast_matrix

plot_contrast_matrix(contrast_def=activation, design_matrix=design_matrix)

# %%
# Below, we compute the :term:`'estimated effect'<Parameter Estimate>`.
# It is in :term:`BOLD` signal unit, but has no statistical guarantees,
# because it does not take into account the associated variance.

eff_map = fmri_glm.compute_contrast(activation, output_type="effect_size")

# %%
# In order to get statistical significance, we form a t-statistic, and
# directly convert it into z-scale. The z-scale means that the values
# are scaled to match a standard Gaussian distribution (mean=0,
# variance=1), across voxels, if there were no effects in the data.

z_map = fmri_glm.compute_contrast(activation, output_type="z_score")

# %%
# Plot thresholded z scores map
# -----------------------------
#
# We display it on top of the average
# functional image of the series (could be the anatomical image of the
# subject).  We use arbitrarily a threshold of 3.0 in z-scale. We'll
# see later how to use corrected thresholds. We will show 3
# axial views, with display_mode='z' and cut_coords=3.

plot_stat_map(
    z_map,
    bg_img=mean_img,
    threshold=3.0,
    display_mode="z",
    cut_coords=3,
    black_bg=True,
    title="Listening greater than rest (Z>3)",
)
plt.show()

# %%
# Statistical significance testing. One should worry about the
# statistical validity of the procedure: here we used an arbitrary
# threshold of 3.0 but the threshold should provide some guarantees on
# the risk of false detections (aka type-1 errors in statistics).
# One suggestion is to control the false positive rate
# (:term:`fpr<FPR correction>`, denoted by alpha)
# at a certain level, e.g. 0.001: this means that there is 0.1% chance
# of declaring an inactive :term:`voxel`, active.

from nilearn.glm import threshold_stats_img

_, threshold = threshold_stats_img(
    z_map, alpha=0.001, height_control="fpr", two_sided=False
)
print(f"Uncorrected p<0.001 threshold: {threshold:.3f}")
plot_stat_map(
    z_map,
    bg_img=mean_img,
    threshold=threshold,
    display_mode="z",
    cut_coords=3,
    black_bg=True,
    title="Listening greater than rest (p<0.001)",
)
plt.show()

# %%
# The problem is that with this you expect 0.001 * n_voxels to show up
# while they're not active --- tens to hundreds of voxels. A more
# conservative solution is to control the family wise error rate,
# i.e. the probability of making only one false detection, say at
# 5%. For that we use the so-called Bonferroni correction.

_, threshold = threshold_stats_img(
    z_map, alpha=0.05, height_control="bonferroni", two_sided=False
)
print(f"Bonferroni-corrected, p<0.05 threshold: {threshold:.3f}")
plot_stat_map(
    z_map,
    bg_img=mean_img,
    threshold=threshold,
    display_mode="z",
    cut_coords=3,
    black_bg=True,
    title="Listening greater than rest (p<0.05, corrected)",
)
plt.show()

# %%
# This is quite conservative indeed!  A popular alternative is to
# control the expected proportion of
# false discoveries among detections. This is called the False
# discovery rate.

_, threshold = threshold_stats_img(
    z_map, alpha=0.05, height_control="fdr", two_sided=False
)
print(f"False Discovery rate = 0.05 threshold: {threshold:.3f}")
plot_stat_map(
    z_map,
    bg_img=mean_img,
    threshold=threshold,
    display_mode="z",
    cut_coords=3,
    black_bg=True,
    title="Listening greater than rest (fdr=0.05)",
)
plt.show()

# %%
# Finally people like to discard isolated voxels (aka "small
# clusters") from these images. It is possible to generate a
# thresholded map with small clusters removed by providing a
# cluster_threshold argument. Here clusters smaller than 10 voxels
# will be discarded.

clean_map, threshold = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control="fdr",
    cluster_threshold=10,
    two_sided=False,
)
plot_stat_map(
    clean_map,
    bg_img=mean_img,
    threshold=threshold,
    display_mode="z",
    cut_coords=3,
    black_bg=True,
    title="Listening greater than rest (fdr=0.05), clusters > 10 voxels",
)
plt.show()


# %%
# We can save the effect and zscore maps to the disk.
z_map.to_filename(output_dir / "listening_gt_rest_z_map.nii.gz")
eff_map.to_filename(output_dir / "listening_gt_rest_eff_map.nii.gz")

# %%
# We can furthermore extract and report the found positions in a table.

from nilearn.reporting import get_clusters_table

table = get_clusters_table(
    z_map, stat_threshold=threshold, cluster_threshold=20
)
table

# %%
# This table can be saved for future use.

table.to_csv(output_dir / "table.csv")

# %%
# Performing an F-test
# --------------------
#
# "listening > rest" is a typical t test: condition versus baseline.
# Another popular type of test is an F test in which
# one seeks whether a certain combination of conditions
# (possibly two-, three- or higher-dimensional)
# explains a significant proportion of the signal.
# Here one might for instance test which voxels are well
# explained by the combination of more active and less active than rest.

# %%
# Specify the :term:`contrast` and compute the corresponding map.
# Actually, the :term:`contrast` specification is done exactly the same way
# as for t-contrasts.

z_map = fmri_glm.compute_contrast(
    activation, output_type="z_score", stat_type="F"
)

# %%
# Note that the statistic has been converted to a z-variable, which
# makes it easier to represent it.

clean_map, threshold = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control="fdr",
    cluster_threshold=10,
    two_sided=False,
)
plot_stat_map(
    clean_map,
    bg_img=mean_img,
    threshold=threshold,
    display_mode="z",
    cut_coords=3,
    black_bg=True,
    title="Effects of interest (fdr=0.05), clusters > 10 voxels",
    cmap="black_red_r",
    symmetric_cbar=False,
)
plt.show()

# %%
# Oops, there is a lot of non-neural signal in there (ventricles, arteries)...
