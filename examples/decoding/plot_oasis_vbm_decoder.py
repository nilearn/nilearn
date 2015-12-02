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
smoothing applied to the data and the number of features selected by the
Anova step should be set by nested cross-validation, as they impact
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
from scipy import linalg
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker

n_subjects = 100  # more subjects requires more memory

# XXX remove this
data_dir = '/home/ahoyosid/NEUROSPIN/DATA/nilearn_data'

### Load Oasis dataset ########################################################
oasis_dataset = datasets.fetch_oasis_vbm(data_dir=data_dir, n_subjects=n_subjects)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
age = oasis_dataset.ext_vars['age'].astype(float)

# print basic information on the dataset
print('First gray-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.gray_matter_maps[0])  # 3D data
print('First white-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.white_matter_maps[0])  # 3D data

### Preprocess data ###########################################################
nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=2,
                           memory='nilearn_cache')

gm_maps_masked = nifti_masker.fit_transform(gray_matter_map_filenames)
n_samples, n_features = gm_maps_masked.shape
print("%d samples, %d features" % (n_subjects, n_features))

### Prediction with Decoder ###################################################
# remove features with too low between-subject variance
gm_maps_masked = nifti_masker.fit_transform(gray_matter_map_filenames)
gm_maps_masked[:, gm_maps_masked.var(0) < 0.01] = 0.
# final masking
niimgs = nifti_masker.inverse_transform(gm_maps_masked)
gm_maps_masked = nifti_masker.fit_transform(niimgs)
n_samples, n_features = gm_maps_masked.shape

from nilearn.decoding import Decoder
decoder = Decoder(estimator='ridge_regression', mask=nifti_masker,
                  screening_percentile=2, n_jobs=1)

### Fit and predict
decoder.fit(niimgs, age)
age_pred = decoder.predict(niimgs)

# Visualization
# Look at the decoder's discriminating weights
weight_img = decoder.coef_img_['beta']
prediction_accuracy = np.mean(decoder.cv_scores_)

print("=== DECODER ===")
print("Prediction accuracy: %f" % prediction_accuracy)
print("")

# Create the figure
from nilearn.plotting import plot_stat_map, show
bg_filename = gray_matter_map_filenames[0]
z_slice = 0
from nilearn.image.resampling import coord_transform
affine = weight_img.get_affine()
_, _, k_slice = coord_transform(0, 0, z_slice, linalg.inv(affine))
k_slice = np.round(k_slice)

fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

weight_slice_data = weight_img.get_data()[..., k_slice]
vmax = max(-np.min(weight_slice_data), np.max(weight_slice_data)) * 0.5
display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=vmax)
display.title('decoder weights', y=1.2)

show()
