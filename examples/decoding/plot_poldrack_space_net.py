"""
SpaceNet on Jimura et al "mixed gambles" dataset.
==================================================

The segmenting power of SpaceNet is quiete visible here.
"""
# author: DOHMATOB Elvis Dopgima,
#         GRAMFORT Alexandre


### Load data ################################################################
import numpy as np
import nibabel
from scipy import ndimage
from nilearn.datasets import fetch_mixed_gambles
data = fetch_mixed_gambles(n_subjects=16)
zmaps = []
object_category = []
mask = []
for zmap_fname in data.zmaps:
    # load subject data
    img = nibabel.load(zmap_fname)
    this_X = img.get_data()
    affine = img.get_affine()
    finite_mask = np.all(np.isfinite(this_X), axis=-1)
    this_mask = np.logical_and(np.all(this_X != 0, axis=-1),
                               finite_mask)
    this_y = np.array([np.arange(1, 9)] * 6).ravel()

    # gain levels
    if len(this_y) != this_X.shape[-1]:
        raise RuntimeError("%s: Expecting %i volumes, got %i!" % (
            zmap_fname, len(this_y), this_X.shape[-1]))

    # standardize subject data
    this_X -= this_X.mean(axis=-1)[..., np.newaxis]
    std = this_X.std(axis=-1)
    std[std == 0] = 1
    this_X /= std[..., np.newaxis]

    # commit subject data
    zmaps.append(this_X)
    object_category.extend(this_y)
    mask.append(this_mask)
object_category = np.array(object_category)
zmaps = np.concatenate(zmaps, axis=-1)
mask = np.sum(mask, axis=0) > .5 * len(data.zmaps)
mask = np.logical_and(mask, np.all(np.isfinite(zmaps), axis=-1))
zmaps = zmaps[mask, :].T
tmp = np.zeros(list(mask.shape) + [len(zmaps)])
tmp[mask, :] = zmaps.T
mask_img = nibabel.Nifti1Image(mask.astype(np.int), affine)
zmaps = nibabel.four_to_three(nibabel.Nifti1Image(tmp, affine))


### Fit TV-L1 #################################################################
from nilearn.decoding import SpaceNetRegressor
decoder = SpaceNetRegressor(mask=mask_img, penalty="tv-l1",
                            eps=1e-1,  # prefer large alphas
                            memory="cache")
decoder.fit(zmaps, object_category)  # fit

### Visualize TV-L1 weights
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img
plot_stat_map(mean_img(decoder.coef_img_), title="tv-l1", display_mode="yz",
              cut_coords=[20, -2])


### Fit Graph-Net ##########################################################
decoder = SpaceNetRegressor(mask=mask_img, penalty="graph-net",
                            eps=1e-1,  # prefer large alphas
                            memory="cache")
decoder.fit(zmaps, object_category)  # fit

### Visualize Graph-Net weights
plot_stat_map(mean_img(decoder.coef_img_), title="graph-net",
              display_mode="yz", cut_coords=[20, -2])


plt.show()
