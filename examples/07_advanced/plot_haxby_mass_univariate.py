"""
Massively univariate analysis of face vs house recognition
==========================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to determine whether or not it behaves differently under a "face
viewing" condition and a "house viewing" condition.
We consider the mean image per session and per condition.
Otherwise, the observations cannot be exchanged at random because
a time dependence exists between observations within a same session
(see [1] for more detailed explanations).

The example shows the small differences that exist between
Bonferroni-corrected p-values and family-wise corrected p-values obtained
from a permutation test combined with a max-type procedure [2].
Bonferroni correction is a bit conservative, as revealed by the presence of
a few false negative.

.. include:: ../../../examples/masker_note.rst

References
----------
[1] Winkler, A. M. et al. (2014).
    Permutation inference for the general linear model. Neuroimage.

[2] Anderson, M. J. & Robinson, J. (2001).
    Permutation tests for linear models.
    Australian & New Zealand Journal of Statistics, 43(1), 75-88.
    (http://avesbiodiv.mncn.csic.es/estadistica/permut2.pdf)

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014

##############################################################################
# Load Haxby dataset
from nilearn import datasets, image

haxby_dataset = datasets.fetch_haxby(subjects=[2])

# print basic information on the dataset
print(f"Mask nifti image (3D) is located at: {haxby_dataset.mask}")
print(f"Functional nifti image (4D) is located at: {haxby_dataset.func[0]}")

##############################################################################
# Restrict to faces and houses
import numpy as np
import pandas as pd

labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = labels["labels"]
categories = conditions.unique()
conditions_encoded = np.zeros_like(conditions)
for c, category in enumerate(categories):
    conditions_encoded[conditions == category] = c
sessions = labels["chunks"]
condition_mask = conditions.isin(["face", "house"])
conditions_encoded = conditions_encoded[condition_mask]

##############################################################################
# Mask data
from nilearn.image import index_img
from nilearn.maskers import NiftiMasker

mask_filename = haxby_dataset.mask

nifti_masker = NiftiMasker(
    smoothing_fwhm=8,
    mask_img=mask_filename,
    memory="nilearn_cache",  # cache options
    memory_level=1,
)
func_filename = haxby_dataset.func[0]
func_reduced = index_img(func_filename, condition_mask)
fmri_masked = nifti_masker.fit_transform(func_reduced)

# We consider the mean image per session and per condition.
# Otherwise, the observations cannot be exchanged at random because
# a time dependence exists between observations within a same session.
n_sessions = np.unique(sessions).size
conditions_per_session = 2
grouped_fmri_masked = np.empty(
    (conditions_per_session * n_sessions, fmri_masked.shape[1])
)
grouped_conditions_encoded = np.empty((conditions_per_session * n_sessions, 1))

for s in range(n_sessions):
    session_mask = sessions[condition_mask] == s
    session_house_mask = np.logical_and(
        session_mask, conditions[condition_mask] == "house"
    )
    session_face_mask = np.logical_and(
        session_mask, conditions[condition_mask] == "face"
    )
    grouped_fmri_masked[2 * s] = fmri_masked[session_house_mask].mean(0)
    grouped_fmri_masked[2 * s + 1] = fmri_masked[session_face_mask].mean(0)
    grouped_conditions_encoded[2 * s] = conditions_encoded[session_house_mask][
        0
    ]
    grouped_conditions_encoded[2 * s + 1] = conditions_encoded[
        session_face_mask
    ][0]

##############################################################################
# Perform massively univariate analysis with permuted OLS
#
# We use a two-sided t-test to compute p-values, but we keep trace of the
# effect sign to add it back at the end and thus observe the signed effect
from nilearn.mass_univariate import permuted_ols

# Note that an intercept as a covariate is used by default
neg_log_pvals, t_scores_original_data, _ = permuted_ols(
    grouped_conditions_encoded,
    grouped_fmri_masked,
    n_perm=10000,
    two_sided_test=True,
    verbose=1,  # display progress bar
    n_jobs=1,  # can be changed to use more CPUs
)
signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    signed_neg_log_pvals
)

##############################################################################
# scikit-learn F-scores for comparison
#
# F-test does not allow to observe the effect sign (pure two-sided test)
from sklearn.feature_selection import f_regression

# f_regression implicitly adds intercept
_, pvals_bonferroni = f_regression(
    grouped_fmri_masked, grouped_conditions_encoded
)
pvals_bonferroni *= fmri_masked.shape[1]
pvals_bonferroni[np.isnan(pvals_bonferroni)] = 1
pvals_bonferroni[pvals_bonferroni > 1] = 1
neg_log_pvals_bonferroni = -np.log10(pvals_bonferroni)
neg_log_pvals_bonferroni_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_bonferroni
)

##############################################################################
# Visualization
import matplotlib.pyplot as plt
from nilearn.image import get_data
from nilearn.plotting import plot_stat_map, show

# Use the fmri mean image as a surrogate of anatomical data
mean_fmri_img = image.mean_img(func_filename)

threshold = -np.log10(0.1)  # 10% corrected

vmax = min(signed_neg_log_pvals.max(), neg_log_pvals_bonferroni.max())

# Plot thresholded p-values map corresponding to F-scores
display = plot_stat_map(
    neg_log_pvals_bonferroni_unmasked,
    mean_fmri_img,
    threshold=threshold,
    cmap=plt.cm.RdBu_r,
    display_mode="z",
    cut_coords=[-1],
    vmax=vmax,
)

neg_log_pvals_bonferroni_data = get_data(neg_log_pvals_bonferroni_unmasked)
n_detections = (neg_log_pvals_bonferroni_data > threshold).sum()
title = (
    "Negative $\\log_{10}$ p-values"
    "\n(Parametric two-sided F-test"
    "\n+ Bonferroni correction)"
    f"\n{n_detections} detections"
)

display.title(title, y=1.1)

# Plot permutation p-values map
display = plot_stat_map(
    signed_neg_log_pvals_unmasked,
    mean_fmri_img,
    threshold=threshold,
    cmap=plt.cm.RdBu_r,
    display_mode="z",
    cut_coords=[-1],
    vmax=vmax,
)

n_detections = (np.abs(signed_neg_log_pvals) > threshold).sum()
title = (
    "Negative $\\log_{10}$ p-values"
    "\n(Non-parametric two-sided test"
    "\n+ max-type correction)"
    f"\n{n_detections} detections"
)

display.title(title, y=1.1)

show()
