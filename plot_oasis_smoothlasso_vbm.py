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
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>

import numpy as np
import matplotlib.pyplot as plt
import nibabel
from sklearn.externals.joblib import Memory
from nilearn import datasets
from nilearn.input_data import NiftiMasker

n_subjects = 100   # more subjects requires more memory
memory = Memory("cache")

### Load Oasis dataset ########################################################
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)

### Preprocess data ###########################################################
nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=2,
    memory=memory)  # cache options
# remove features with too low between-subject variance
gm_maps_masked = nifti_masker.fit_transform(dataset_files.gray_matter_maps)
gm_maps_masked[:, gm_maps_masked.var(0) < 0.01] = 0.
# final masking
new_images = nifti_masker.inverse_transform(gm_maps_masked)
gm_maps_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = gm_maps_masked.shape
mask = nifti_masker.mask_img_.get_data().astype(np.bool)
print n_samples, "subjects, ", n_features, "features"

import os
from nilearn.sparse_models.cv import SmoothLassoRegressorCV, plot_cv_scores
n_jobs = int(os.environ.get("N_JOBS", 1))
slcv = SmoothLassoRegressorCV(verbose=1, n_jobs=n_jobs, memory=memory,
                              mask=mask, screening_percentile=5)

### Fit and predict
slcv.fit(gm_maps_masked, age)
age_pred = slcv.predict(gm_maps_masked).ravel()

### Visualisation
### Look at the SLCV's discriminating weights
# reverse masking
weight_niimg = nifti_masker.inverse_transform(slcv.coef_)

# We use a masked array so that the voxels at '-1' are transparent
weights = np.ma.masked_array(weight_niimg.get_data(),
                             weight_niimg.get_data() == 0)

### Create the figure
background_img = nibabel.load(dataset_files.gray_matter_maps[0]).get_data()
picked_slice = 36
plt.figure(figsize=(5.5, 5.5))
data_for_plot = weights[:, :, picked_slice, 0]
vmax = max(np.min(data_for_plot), np.max(data_for_plot)) * 0.5
plt.imshow(np.rot90(background_img[:, :, picked_slice]), cmap=plt.cm.gray,
          interpolation='nearest')
im = plt.imshow(np.rot90(data_for_plot), cmap=plt.cm.Spectral_r,
                interpolation='nearest', vmin=-vmax, vmax=vmax)
plt.axis('off')
plt.colorbar(im)
plt.title('S-LASSO weights')
plot_cv_scores(slcv, errorbars=False)
plt.figure()
linewidth = 3
ax1 = plt.subplot('211')
ax1.plot(age, label="True age", linewidth=linewidth)
ax1.plot(age_pred, '--', c="g", label="Fitted age", linewidth=linewidth)
ax1.set_ylabel("age")
plt.legend(loc="best")
ax2 = plt.subplot("212")
ax2.plot(age - age_pred, label="True age - fitted age", linewidth=linewidth)
ax2.set_xlabel("subject")
plt.legend(loc="best")
plt.show()
