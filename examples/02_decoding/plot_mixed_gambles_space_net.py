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
import time

start_time = time.time()
tv_l1 = SpaceNetRegressor(mask=mask_filename, penalty="tv-l1",
                          eps=1e-1,  # prefer large alphas
                          memory="nilearn_cache")

tv_l1.fit(zmap_filenames, behavioral_target)
print("Space Net with TV-L1 penalty was fitted in {} seconds".format(int(time.time() - start_time)))

# Visualize TV-L1 weights
# ------------------------
from nilearn.plotting import plot_stat_map, show
plot_stat_map(tv_l1.coef_img_, title="tv-l1", display_mode="yz",
              cut_coords=[20, -2])

##########################################################################
# Fit Graph-Net
# --------------
start_time = time.time()
graph_net = SpaceNetRegressor(mask=mask_filename, penalty="graph-net",
                              eps=1e-1,  # prefer large alphas
                              memory="nilearn_cache")
graph_net.fit(zmap_filenames, behavioral_target)
print("Space Net with graph-net penalty was fitted in {} seconds".format(int(time.time() - start_time)))

# Visualize Graph-Net weights
# ----------------------------
plot_stat_map(graph_net.coef_img_, title="graph-net", display_mode="yz",
              cut_coords=[20, -2])

##########################################################################
# Fit fREM
# ----------
# We compare both of these models to a pipeline ensembling many models
from nilearn.decoding import fREMRegressor
start_time = time.time()
fREM = fREMRegressor('svr', clustering_percentile=10,
                     screening_percentile=20, cv=10)

fREM.fit(zmap_filenames, behavioral_target)
print("fREM was fitted in {} seconds".format(int(time.time() - start_time)))

# Visualize fREM weights
# ----------------------------
plot_stat_map(fREM.coef_img_['beta'], title="fREM", display_mode="yz",
              cut_coords=[20, -2], threshold=.2)

##########################################################################
# We can see that Space Net model yields a sparse coefficients map with
# both penalties, more structured (and thus interpretable) with TV-L1.
#
# The coefficient maps learnt by fREM is not sparse (it has been
# thresholded for display) but is structured as well. Importantly, fREM is
# faster than TV-L1 Space Net (7 times here).
