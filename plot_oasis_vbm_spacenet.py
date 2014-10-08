"""
Voxel-Based Morphometry on Oasis dataset with Space-Net prior
=============================================================

"""
# Authors: Elvis DOHMATOB,
#          Virgile FRITSCH

from nilearn import datasets

n_subjects = 100  # more subjects requires more memory

### Load Oasis dataset ########################################################
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)


### Fit and predict ###########################################################
from nilearn.decoding import SpaceNet
decoder = SpaceNet(memory="cache", screening_percentile=20., verbose=1,
                   smoothing_fwhm=2.)
decoder.fit(dataset_files.gray_matter_maps, age)
coef_img = decoder.coef_img_
age_pred = decoder.predict(dataset_files.gray_matter_maps).ravel()

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
background_img = dataset_files.gray_matter_maps[0]
plot_stat_map(coef_img, background_img, title="Graph-Net weights",
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
