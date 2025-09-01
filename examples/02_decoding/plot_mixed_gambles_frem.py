"""
FREM on Jimura et al "mixed gambles" dataset
============================================

In this example, we use fast ensembling of regularized models (FREM) to
solve a regression problem, predicting the gain level corresponding to each
:term:`beta<Beta>` maps regressed from mixed gambles experiment.
:term:`FREM` uses an implicit spatial regularization through fast clustering
and aggregates a high number
of  estimators trained on various splits of the training set, thus returning
a very robust decoder at a lower computational cost than other spatially
regularized methods.

To have more details, see: :ref:`frem`.

See the :ref:`dataset description <mixed_gamble_maps>`
for more information on the data used in this example.
"""

# %%
# Load the data from the Jimura mixed-gamble experiment
# -----------------------------------------------------
from nilearn.datasets import fetch_mixed_gambles

data = fetch_mixed_gambles(n_subjects=16)

zmap_filenames = data.zmaps
behavioral_target = data.gain.to_numpy().ravel()
mask_filename = data.mask_img

# %%
# Fit FREM
# --------
# We compare both of these models to a pipeline ensembling many models
#
import warnings

from sklearn.exceptions import ConvergenceWarning

from nilearn.decoding import FREMRegressor

frem = FREMRegressor("svr", cv=10, standardize="zscore_sample", verbose=1)

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    frem.fit(zmap_filenames, behavioral_target)

# %%
# Visualize FREM weights
# ----------------------

from nilearn.plotting import plot_stat_map, show

plot_stat_map(
    frem.coef_img_["beta"],
    title="FREM",
    display_mode="yz",
    cut_coords=[20, -2],
    threshold=0.2,
)

show()

# %%
# We can observe that the coefficients map learnt
# by :term:`FREM` is structured,
# due to the spatial regularity imposed by working on clusters and model
# ensembling. Although these maps have been thresholded for display, they are
# not sparse (i.e. almost all voxels have non-zero coefficients).
#
# .. seealso::
#
#   :ref:`other example
#   <sphx_glr_auto_examples_02_decoding_plot_haxby_frem.py>`
#   using FREM, and related :ref:`section of user guide <frem>`.
#

# %%
# Example use of TV-L1 SpaceNet
# -----------------------------
# :ref:`SpaceNet<space_net>` is another method available in Nilearn to decode
# with spatially sparse models. Depending on the penalty that is used,
# it yields either very structured maps (TV-L1) or unstructured maps
# (graph_net). Because of their heavy computational costs, these methods are
# not demonstrated on this example but you can try them easily if you have a
# few minutes. Example code is included below.
#

from nilearn.decoding import SpaceNetRegressor

# We use the regressor object since the task is to predict a continuous
# variable (gain of the gamble).

tv_l1 = SpaceNetRegressor(
    mask=mask_filename,
    penalty="tv-l1",
    eps=1e-1,  # prefer large alphas
    memory="nilearn_cache",
    n_jobs=2,
)
# tv_l1.fit(zmap_filenames, behavioral_target)
# plot_stat_map(tv_l1.coef_img_, title="TV-L1", display_mode="yz",
#               cut_coords=[20, -2])

# sphinx_gallery_dummy_images=1
