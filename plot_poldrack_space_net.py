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
mask_img = nibabel.Nifti1Image(mask.astype(np.int), affine)
X_train = nibabel.Nifti1Image(img_data, affine)
y_train = y

### Fit and predict ##########################################################
import os
from nilearn.decoding import SpaceNetRegressor
penalty = "TV-L1"
l1_ratio = .3
alpha = None
decoder = SpaceNetRegressor(memory=mem, mask=mask_img, verbose=2,
                            n_jobs=int(os.environ.get("N_JOBS", 1)),
                            l1_ratio=l1_ratio, penalty=penalty, alpha=alpha,
                            screening_percentile=100., tol=1e-8,
                            max_iter=1000)
decoder.fit(X_train, y_train)  # fit
coef_niimg = decoder.coef_img_
coef_niimg.to_filename('poldrack_%s(l1_ratio=%g, alpha=%s)_weights.nii' % (
        penalty, l1_ratio, alpha))

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
background_img = mean_img(X_train)
for penalty, decoder in decoders.iteritems():
    coef_img = mean_img(decoder.coef_img_)
    plot_stat_map(coef_img, background_img, title=penalty, display_mode="yz",
                  cut_coords=[20, -2])
plt.show()
