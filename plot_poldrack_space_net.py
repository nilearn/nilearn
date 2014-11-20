import numpy as np
from joblib import Memory

### Load data ################################################################
from load_poldrack import load_gain_poldrack
mem = Memory(cachedir='cache', verbose=3)
X, y, _, mask, affine = mem.cache(load_gain_poldrack)(smooth=0)
img_data = np.zeros(list(mask.shape) + [len(X)])
img_data[mask, :] = X.T

# prepare input data for learner
import nibabel
n_samples = img_data.shape[-1]
n_samples_train = n_samples * 8 / 10
mask_img = nibabel.Nifti1Image(mask.astype(np.int), affine)
X_train = nibabel.Nifti1Image(img_data[:, :, :, :n_samples_train], affine)
y_train = y[:n_samples_train]
X_test = nibabel.Nifti1Image(img_data[:, :, :, n_samples_train:], affine)
y_test = y[n_samples_train:]

### Fit and predict ##########################################################
from nilearn.decoding import SpaceNetRegressor
penalties = ["smooth-lasso", "TV-L1"]
decoders = {}
for penalty in penalties:
    decoder = SpaceNetRegressor(memory=mem, mask=mask_img, verbose=2,
                                penalty=penalty)
    decoder.fit(X_train, y_train)  # fit
    decoders[penalty] = decoder

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
background_img = mean_img(X_train)
for penalty, decoder in decoders.iteritems():
    plot_stat_map(mean_img(decoder.coef_img_), background_img, title=penalty,
                  display_mode="yz", cut_coords=[20, -2])
plt.show()
