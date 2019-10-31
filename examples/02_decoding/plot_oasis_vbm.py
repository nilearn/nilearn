"""
Voxel-Based Morphometry on Oasis dataset
========================================

This example uses Voxel-Based Morphometry (VBM) to study the relationship
between aging and gray matter density.

The data come from the `OASIS <http://www.oasis-brains.org/>`_ project.
If you use it, you need to agree with the data usage agreement available
on the website.

It has been run through a standard VBM pipeline (using SPM8 and
NewSegment) to create VBM maps, which we study here.

Predictive modeling analysis: VBM bio-markers of aging?
--------------------------------------------------------

We run a standard SVM-ANOVA nilearn pipeline to predict age from the VBM
data. We use only 100 subjects from the OASIS dataset to limit the memory
usage.

Note that for an actual predictive modeling study of aging, the study
should be ran on the full set of subjects. Also, all parameters should be set
by cross-validation. This includes the smoothing applied to the data and the
number of features selected by the ANOVA step. Indeed, even these
data-preparation parameter impact significantly the prediction score.


Also, parameters such as the
smoothing should be applied to the data and the number of features selected
by the ANOVA step should be set by nested cross-validation, as they impact
significantly the prediction score.

Brain mapping with mass univariate
-----------------------------------

SVM weights are very noisy, partly because heavy smoothing is detrimental
for the prediction here. A standard analysis using mass-univariate GLM
(here permuted to have exact correction for multiple comparisons) gives a
much clearer view of the important regions.

____

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
#          Gael Varoquaux, Apr 2014
#          Andres Hoyos-Idrobo, Apr 2017

######################################################################
# Loading the data
# --------------------

# First, we load the data
import numpy as np
from nilearn import datasets
n_subjects = 200  # increase this number if you have more RAM on your box
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
gm_imgs = dataset_files.gray_matter_maps

# Split data into training set and test set
from sklearn.model_selection import train_test_split
gm_imgs_train, gm_imgs_test, age_train, age_test = train_test_split(
    gm_imgs, age, train_size=.6, random_state=0)
######################################################################
# Preprocess the mask
# --------------------

# First we use the :class:`nilearn.input_data.NiftiMasker` to extract 
# fMRI data on a mask and convert it to data series.
from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=2,
                           memory='nilearn_cache')
gm_maps_masked = nifti_masker.fit_transform(gm_imgs_train)
##########################################################################
# The features with too low between-subject variance are removed using
# :class:`sklearn.feature_selection.VarianceThreshold`.
from sklearn.feature_selection import VarianceThreshold
variance_threshold = VarianceThreshold(threshold=.01)
gm_maps_thresholded = variance_threshold.fit_transform(gm_maps_masked)
gm_maps_masked = variance_threshold.inverse_transform(gm_maps_thresholded)

# Then we convert the data back to the mask image in order to use it for 
# decoding process
mask = nifti_masker.inverse_transform(variance_threshold.get_support())
###############################################################################
# Training the decoder
# ---------------------

# To save time (because these are anat images with many voxels), we include
# only the 5-percent voxels most correlated with the age variable to fit.
# Also, we set memory_level=2 so that more of the intermediate computations
# are cached. Also, you may pass and n_jobs=<some_high_value> to the
# DecoderRegressor class, to take advantage of a multi-core system. We also
# want to set mask hyperparameter to be the mask we just obtained above.
from nilearn.decoding import DecoderRegressor
decoder = DecoderRegressor(estimator='svr', mask=mask,
                           screening_percentile=5,
                           memory_level=2,
                           memory='nilearn_cache')  # cache options
# Fit and predict with the decoder
decoder.fit(gm_imgs_train, age_train)

# Sort test data for better visualization (trend, etc.)
perm = np.argsort(age_test)[::-1]
age_test = age_test[perm]
gm_imgs_test = np.array(gm_imgs_test)[perm]
age_pred = decoder.predict(gm_imgs_test)

prediction_score = np.mean(decoder.cv_scores_['beta'])

print("=== DECODER ===")
print("explained variance for the cross-validation: %f" % prediction_score)
print("")

######################################################################
# Visualization
# --------------
weight_img = decoder.coef_img_['beta']

# Create the figure
from nilearn.plotting import plot_stat_map, show
bg_filename = gm_imgs[0]

display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[-6],
                        title="Decoder r2: %g" % prediction_score)

show()

######################################################################
# Visualize the quality of predictions
# -------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4.5))
plt.suptitle("Decoder: Mean Absolute Error %.2f years" % prediction_score)
linewidth = 3
plt.plot(age_test, label="True age", linewidth=linewidth)
plt.plot(age_pred, '--', c="g", label="Predicted age", linewidth=linewidth)
plt.ylabel("age")
plt.xlabel("subject")
plt.legend(loc="best")
######################################################################
plt.figure(figsize=(6, 4.5))
plt.plot(age_test - age_pred, label="True age - predicted age",
         linewidth=linewidth)
plt.xlabel("subject")
plt.legend(loc="best")
###############################################################################
# Comparing to massi-univariate analysis
# ---------------------------------------
print("Massively univariate model")

# First, we have to use the mask that we obtained after the variance
# thresholding.
nifti_masker = NiftiMasker(mask_img=mask)
gm_imgs_train_masked = nifti_masker.fit_transform(gm_imgs_train)

# Statistical inference
from nilearn.mass_univariate import permuted_ols
neg_log_pvals, t_scores_original_data, _ = permuted_ols(
    age_train, gm_imgs_train_masked,  # + intercept as a covariate by default
    n_perm=2000,  # 1,000 in the interest of time; 10000 would be better
    n_jobs=1)  # can be changed to use more CPUs
signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    signed_neg_log_pvals)

# Show results
threshold = -np.log10(0.1)  # 10% corrected
fig = plt.figure(figsize=(5, 6), facecolor='k')
display = plot_stat_map(signed_neg_log_pvals_unmasked, bg_img=bg_filename,
                        threshold=threshold, cmap=plt.cm.RdBu_r,
                        display_mode='z', cut_coords=[-6], figure=fig)
title = 'Negative $\log_{10}$ p-values\n(Non-parametric + max-type correction)'
display.title(title, y=1.2)
n_detections = (signed_neg_log_pvals_unmasked.get_data() > threshold).sum()
print('\n%d detections' % n_detections)

show()
