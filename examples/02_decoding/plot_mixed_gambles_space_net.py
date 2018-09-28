"""
SpaceNet on Jimura et al "mixed gambles" dataset.
==================================================

The segmenting power of SpaceNet is quite visible here.

See also the SpaceNet documentation: :ref:`space_net`.
"""
# author: DOHMATOB Elvis Dopgima,
#         GRAMFORT Alexandre


##########################################################################
# Load the data from the Jimura mixed-gamble experiment
# ------------------------------------------------------
from nilearn.datasets import fetch_mixed_gambles
data = fetch_mixed_gambles(n_subjects=16)

zmap_filenames = data.zmaps
behavioral_target = data.gain
mask_filename = data.mask_img


##########################################################################
# Fit TV-L1
# ----------
# Here we're using the regressor object given that the task is to predict a
# continuous variable, the gain of the gamble.
from nilearn.decoding import SpaceNetRegressor
decoder = SpaceNetRegressor(mask=mask_filename, penalty="tv-l1",
                            eps=1e-1,  # prefer large alphas
                            memory="nilearn_cache")

decoder.fit(zmap_filenames, behavioral_target)

# Visualize TV-L1 weights
# ------------------------
from nilearn.plotting import plot_stat_map, show
plot_stat_map(decoder.coef_img_, title="tv-l1", display_mode="yz",
              cut_coords=[20, -2])


##########################################################################
# Fit Graph-Net
# --------------
decoder = SpaceNetRegressor(mask=mask_filename, penalty="graph-net",
                            eps=1e-1,  # prefer large alphas
                            memory="nilearn_cache")
decoder.fit(zmap_filenames, behavioral_target)

# Visualize Graph-Net weights
# ----------------------------
plot_stat_map(decoder.coef_img_, title="graph-net", display_mode="yz",
              cut_coords=[20, -2])

show()
