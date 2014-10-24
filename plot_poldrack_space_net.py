import numpy as np
from joblib import Memory
from examples.proximal.load_data import load_gain_poldrack

mem = Memory(cachedir='cache', verbose=3)

n_jobs = 1

X, y, subjects, mask, affine = mem.cache(load_gain_poldrack)(smooth=0)

# make X (design matrix) and y (response variable)
img_data = np.zeros(list(mask.shape) + [len(X)])
img_data[mask, :] = X.T

import nibabel
mask_img = nibabel.Nifti1Image(mask.astype(np.int), affine)
X_train = nibabel.Nifti1Image(img_data, affine)
y_train = y

### Fit and predict ##########################################################
import os
from nilearn.decoding import SpaceNetRegressor
decoder = SpaceNetRegressor(memory="cache", mask=mask_img, verbose=2,
                            n_jobs=int(os.environ.get("N_JOBS", 1)))
decoder.fit(X_train, y_train)  # fit
coef_niimg = decoder.coef_img_
coef_niimg.to_filename('poldrack_weights.nii')

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
plt.close('all')
background_img = mean_img(X_train)
background_img.to_filename('poldrack_mean.nii')
slicer = plot_stat_map(coef_niimg, background_img, title="Weights",
                       cut_coords=range(10, 30, 2), display_mode="y")
slicer.add_contours(decoder.mask_img_)
plt.show()
