"""
Deriving spatial maps from group fMRI data using ICA and Dictionary Learning
============================================================================

Various approaches exist to derive spatial maps or networks from
group :term`fMRI` data. The methods extract distributed brain regions that
exhibit similar :term:`BOLD` fluctuations over time. Decomposition
methods allow for generation of many independent maps simultaneously
without the need to provide a priori information (e.g. seeds or priors).

This example will apply two popular decomposition methods, :term:`ICA` and
:term:`Dictionary learning`, to :term:`fMRI` data measured while children
and young adults watch movies. The resulting maps will be visualized using
atlas plotting tools.

:term:`CanICA` is an :term:`ICA` method
for group-level analysis of :term:`fMRI` data.
Compared to other strategies, it brings a well-controlled group model,
as well as a
thresholding algorithm controlling for specificity and sensitivity with
an explicit model of the signal.

The reference paper is :footcite:t:`Varoquaux2010c`.
"""

# %%
# Load brain development :term:`fMRI` dataset
# -------------------------------------------
from nilearn.datasets import fetch_development_fmri

rest_dataset = fetch_development_fmri(n_subjects=30)
func_filenames = rest_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print(f"First functional nifti image (4D) is at: {rest_dataset.func[0]}")


# %%
# Apply :term:`CanICA` on the data
# --------------------------------
# We use "whole-brain-template" as a strategy to compute the mask,
# as this leads to slightly faster and more reproducible results.
# However, the images need to be in :term:`MNI` template space.

import warnings

from sklearn.exceptions import ConvergenceWarning

from nilearn.decomposition import CanICA

canica = CanICA(
    n_components=20,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
    random_state=0,
    mask_strategy="whole-brain-template",
    n_jobs=2,
)
with warnings.catch_warnings():
    # silence warnings about ICA not converging
    # Consider increasing tolerance or the maximum number of iterations.
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    canica.fit(func_filenames)

# Retrieve the independent components in brain space directly
# accessible through attribute `components_img_`.
canica_components_img = canica.components_img_


# %%
# To visualize, we plot the outline of all components on one figure
from nilearn.plotting import plot_prob_atlas

# Plot all ICA components together
plot_prob_atlas(canica_components_img, title="All ICA components")


# %%
# Finally, we plot the map for each :term:`ICA` component separately.
#
# .. note::
#
#   The following code block will generate many figures.
#
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

for i, cur_img in enumerate(iter_img(canica_components_img)):
    plot_stat_map(
        cur_img,
        display_mode="z",
        title=f"IC {int(i)}",
        cut_coords=1,
        vmax=0.05,
        vmin=-0.05,
        colorbar=False,
    )


show()

# %%
# Compare :term:`CanICA` to dictionary learning
# ---------------------------------------------
# :term:`Dictionary learning` is a sparsity based decomposition method
# for extracting spatial maps. It extracts maps that are naturally sparse
# and usually cleaner than :term:`ICA`. Here, we will compare networks built
# with :term:`CanICA` to networks built with :term:`Dictionary learning`.
#
# For more details see :footcite:t:`Mensch2016`.
#


# %%
# Create a dictionary learning estimator
from nilearn.decomposition import DictLearning

dict_learning = DictLearning(
    n_components=20,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
    random_state=0,
    n_epochs=1,
    mask_strategy="whole-brain-template",
    n_jobs=2,
)

dict_learning.fit(func_filenames)

dictlearning_components_img = dict_learning.components_img_


# %%
# Visualize the results
#
# First plot all DictLearning components together
plot_prob_atlas(
    dictlearning_components_img, title="All DictLearning components"
)


# %%
# One plot of each component
#
# .. note::
#
#   The following code block will generate many figures.
#

for i, cur_img in enumerate(iter_img(dictlearning_components_img)):
    plot_stat_map(
        cur_img,
        display_mode="z",
        title=f"Comp {int(i)}",
        cut_coords=1,
        vmax=0.1,
        vmin=-0.1,
        colorbar=False,
    )

# %%
# Estimate explained variance per component and plot using matplotlib
#
# The fitted object ``dict_learning`` can be used
# to calculate the score per component.
scores = dict_learning.score(func_filenames, per_component=True)

# Plot the scores
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.figure(figsize=(4, 4), constrained_layout=True)

positions = np.arange(len(scores))
plt.barh(positions, scores)
plt.ylabel("Component #", size=12)
plt.xlabel("Explained variance", size=12)
plt.yticks(np.arange(20))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

show()

# %%
# .. note::
#
#     To see how to extract subject-level timeseries from regions
#     created using :term:`Dictionary learning`, see :ref:`example Regions
#     extraction using dictionary learning and functional connectomes
#     <sphx_glr_auto_examples_03_connectivity\
#     _plot_extract_regions_dictlearning_maps.py>`.

# %%
# References
# ----------
#
# .. footbibliography::


# sphinx_gallery_dummy_images=5
