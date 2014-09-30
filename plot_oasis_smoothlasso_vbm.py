"""
Voxel-Based Morphometry on Oasis dataset with S-LASSO prior
===========================================================

"""
# Authors: Elvis DOHMATOB,
#          Virgile FRITSCH

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

from nilearn.decoding.space_net import SpaceNet
slcv = SpaceNet(memory=memory, screening_percentile=10, verbose=1,
                mask=nifti_masker, n_jobs=14)

### Fit and predict
slcv.fit(new_images, age)
coef_niimg = slcv.coef_img_
age_pred = slcv.predict(new_images).ravel()

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
background_niimg = nibabel.load(dataset_files.gray_matter_maps[0])
plot_stat_map(coef_niimg, background_niimg, title="S-LASSO weights",
              display_mode="z")

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
