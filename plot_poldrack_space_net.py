import numpy as np
from joblib import Memory

import os
import sys
sys.path.append(
    os.path.join(os.environ["HOME"], "CODE/FORKED/parietal-python"))
from examples.proximal.load_data import load_gain_poldrack

mem = Memory(cachedir='cache', verbose=3)
X, y, _, mask, affine = mem.cache(load_gain_poldrack)(smooth=0)
img_data = np.zeros(list(mask.shape) + [len(X)])
img_data[mask, :] = X.T

import nibabel
mask_img = nibabel.Nifti1Image(mask.astype(np.int), affine)
X_train = nibabel.Nifti1Image(img_data, affine)
y_train = y

### Fit and predict ##########################################################
from nilearn.decoding import SpaceNetRegressor
penalty = "TV-L1"
l1_ratio = .9
alpha = None  # 1. / X_train.shape[-1]
max_iter = 1000
tol = 1e-4
decoder = SpaceNetRegressor(memory=mem, mask=mask_img, verbose=2,
                            n_jobs=int(os.environ.get("N_JOBS", 1)),
                            cv=8, l1_ratio=l1_ratio, penalty=penalty,
                            alpha=alpha, max_iter=max_iter, tol=tol)
decoder.fit(X_train, y_train)  # fit
coef_niimg = decoder.coef_img_
coef_niimg.to_filename('poldrack_%s(l1_ratio=%g, alpha=%s)_weights.nii' % (
        penalty, l1_ratio, alpha))

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
plt.close('all')
background_img = mean_img(X_train)
background_img.to_filename('poldrack_mean.nii')
slicer = plot_stat_map(coef_niimg, background_img, title="Weights",
                       cut_coords=[20, -2], display_mode="yz")
slicer.add_contours(decoder.mask_img_)
plt.show()
