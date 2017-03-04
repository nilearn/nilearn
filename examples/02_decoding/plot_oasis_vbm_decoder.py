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
smoothing should be applied to the data and the number of features selected by
the Anova step should be set by nested cross-validation, as they impact
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
#          Andres Hoyos-Idrobo, Dec 2015
import numpy as np

n_subjects = 100  # more subjects requires more memory

# Load Oasis dataset
from nilearn import datasets
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
age = oasis_dataset.ext_vars['age'].astype(float)

# print basic information on the dataset
print('First gray-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.gray_matter_maps[0])  # 3D data
print('First white-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.white_matter_maps[0])  # 3D data

# Preprocess data
from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=2,
                           memory='nilearn_cache')

gm_maps_masked = nifti_masker.fit_transform(gray_matter_map_filenames)
n_samples, n_features = gm_maps_masked.shape
print("%d samples, %d features" % (n_subjects, n_features))

# Prediction with Decoder
# remove features with too low between-subject variance
gm_maps_masked = nifti_masker.fit_transform(gray_matter_map_filenames)
gm_maps_masked[:, gm_maps_masked.var(0) < 0.01] = 0.
# final masking
niimgs = nifti_masker.inverse_transform(gm_maps_masked)
gm_maps_masked = nifti_masker.fit_transform(niimgs)
n_samples, n_features = gm_maps_masked.shape

from nilearn.decoding import DecoderRegressor
decoder = DecoderRegressor(estimator='svr', mask=nifti_masker,
                           scoring='neg_mean_absolute_error',
                           screening_percentile=5, n_jobs=1)
# Fit and predict with the decoder
decoder.fit(niimgs, age)
age_pred = decoder.predict(niimgs)
# Visualization
weight_img = decoder.coef_img_['beta']
prediction_score = np.mean(decoder.cv_scores_)

print("=== DECODER ===")
print("Cross-validation score: %f" % prediction_score)
print("")
# Create the figure
from nilearn.plotting import plot_stat_map, show
bg_filename = gray_matter_map_filenames[0]

display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[-6],
                        title="Decoder: r2 %g" % prediction_score)

show()
