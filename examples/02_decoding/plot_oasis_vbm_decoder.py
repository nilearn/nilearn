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
should be ran on the full set of subjects. Also, parameters such as the
smoothing should be applied to the data and the number of features selected
by the Anova step should be set by nested cross-validation, as they impact
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

import numpy as np
from nilearn import datasets
n_subjects = 200  # increase this number if you have more RAM on your box
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
age = np.array(age)
gm_imgs = np.array(dataset_files.gray_matter_maps)

# Split data into training set and test set
from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
rng = check_random_state(42)
gm_imgs_train, gm_imgs_test, age_train, age_test = train_test_split(
    gm_imgs, age, train_size=.6, random_state=rng)

# Preprocess the mask: remove features with too low between-subject variance
from sklearn.feature_selection import VarianceThreshold
from nilearn.input_data import NiftiMasker
variance_threshold = VarianceThreshold(threshold=.01)
nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=2,
                           memory='nilearn_cache')
gm_maps_masked = nifti_masker.fit_transform(gm_imgs_train)
gm_maps_thresholded = variance_threshold.fit_transform(gm_maps_masked)
gm_maps_masked = variance_threshold.inverse_transform(gm_maps_thresholded)
gm_imgs_thresholded = nifti_masker.inverse_transform(gm_maps_masked)
nifti_masker.fit(gm_imgs_thresholded)

# To save time (because these are anat images with many voxels), we include
# only the 2-percent voxels most correlated with the age variable to fit.
# Also, we set memory_level=2 so that more of the intermediate computations
# are cached. Also, you may pass and n_jobs=<some_high_value> to the
# DecoderRegressor class, to take advantage of a multi-core system.
from nilearn.decoding import DecoderRegressor
decoder = DecoderRegressor(estimator='svr', mask=nifti_masker, cv=5,
                           screening_percentile=2, n_jobs=1)
# Fit and predict with the decoder
decoder.fit(gm_imgs_train, age_train)
age_pred = decoder.predict(gm_imgs_test)
# Visualization
weight_img = decoder.coef_img_['beta']
prediction_score = np.mean(decoder.cv_scores_)

print("=== DECODER ===")
print("r2 for the cross-validation: %f" % prediction_score)
print("")
# Create the figure
from nilearn.plotting import plot_stat_map, show
bg_filename = gm_imgs[0]

display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[-6],
                        title="Decoder r2: %g" % prediction_score)
# One can also use other scores to measure the performance of the decoder
from sklearn.metrics.scorer import mean_absolute_error
cv_y_pred = decoder.cv_y_pred_
cv_y_true = decoder.cv_y_true_

prediction_score = mean_absolute_error(cv_y_true, cv_y_pred)

print("=== DECODER ===")
print("cross-validation score: %f years" % prediction_score)
print("")


# ### Inference with massively univariate model #################################
# print("Massively univariate model")

# # Statistical inference
# from nilearn.mass_univariate import permuted_ols
# import matplotlib.pyplot as plt
# data = variance_threshold.fit_transform(gm_maps_masked)
# neg_log_pvals, t_scores_original_data, _ = permuted_ols(
#     age, data,  # + intercept as a covariate by default
#     n_perm=2000,  # 1,000 in the interest of time; 10000 would be better
#     n_jobs=1)  # can be changed to use more CPUs
# signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
# signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
#     variance_threshold.inverse_transform(signed_neg_log_pvals))

# # Show results
# threshold = -np.log10(0.1)  # 10% corrected

# fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

# display = plot_stat_map(signed_neg_log_pvals_unmasked, bg_img=bg_filename,
#                         threshold=threshold, cmap=plt.cm.RdBu_r,
#                         display_mode='z', cut_coords=[z_slice],
#                         figure=fig)
# title = ('Negative $\log_{10}$ p-values'
#          '\n(Non-parametric + max-type correction)')
# display.title(title, y=1.2)

# n_detections = (signed_neg_log_pvals_unmasked.get_data() > threshold).sum()
# print('\n%d detections' % n_detections)

show()
