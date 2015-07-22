"""
SpaceNet on Jimura et al "mixed gambles" dataset.
==================================================

The segmenting power of SpaceNet is quite visible here.
"""
# author: DOHMATOB Elvis Dopgima,
#         GRAMFORT Alexandre


### Load data ################################################################
import numpy as np
import nibabel
from scipy import ndimage
from nilearn.datasets import fetch_mixed_gambles
data = fetch_mixed_gambles(n_subjects=16, return_raw_data=False)
zmaps, object_category, mask_img = data.zmaps, data.gain, data.mask_img


### Fit TV-L1 #################################################################
from nilearn.decoding import SpaceNetRegressor
decoder = SpaceNetRegressor(mask=mask_img, penalty="tv-l1",
                            eps=1e-1,  # prefer large alphas
                            memory="cache")
decoder.fit(zmaps, object_category)  # fit

### Visualize TV-L1 weights
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
plot_stat_map(decoder.coef_img_, title="tv-l1", display_mode="yz",
              cut_coords=[20, -2])


### Fit Graph-Net ##########################################################
decoder = SpaceNetRegressor(mask=mask_img, penalty="graph-net",
                            eps=1e-1,  # prefer large alphas
                            memory="cache")
decoder.fit(zmaps, object_category)  # fit

### Visualize Graph-Net weights
plot_stat_map(decoder.coef_img_, title="graph-net",
              display_mode="yz", cut_coords=[20, -2])


plt.show()
