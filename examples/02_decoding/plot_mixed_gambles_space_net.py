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

# The choices of regularization terms `alphas` are pre-trained by
# cross-validation to save computation time (n_subjects=16) 768 z-maps
alphas = [51.35, 106.90, 57.24, 118.87, 55.72, 57.33, 80.49, 51.29]
##########################################################################
# Fit TV-L1
# ----------
# Here we're using the regressor object given that the task is to predict a
# continuous variable, the gain of the gamble.
from nilearn.decoding import SpaceNetRegressor

# Cross-validation folds are set to 3 to save computation time
decoder = SpaceNetRegressor(mask=mask_filename, penalty="tv-l1",
                            alphas=alphas, cv=3,
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

# We use `alphas` for "graph-net" solver which are pre-trained on jimura z-maps
# of 16 subjects, total 768 z-maps. This is to save computation time.

alphas = [307.85, 384.19, 443.21, 71.26, 200.25, 266.11, 289.29, 238.08]
cv = 3

decoder = SpaceNetRegressor(mask=mask_filename, penalty="graph-net",
                            alphas=alphas, cv=3,
                            eps=1e-1,  # prefer large alphas
                            memory="nilearn_cache")
decoder.fit(zmap_filenames, behavioral_target)

# Visualize Graph-Net weights
# ----------------------------
plot_stat_map(decoder.coef_img_, title="graph-net", display_mode="yz",
              cut_coords=[20, -2])

show()
