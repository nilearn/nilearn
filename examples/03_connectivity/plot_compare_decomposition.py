"""
Deriving spatial maps from group fMRI data using ICA and Dictionary Learning
============================================================================

Various approaches exist to derive spatial maps or networks from
group fmr data. The methods extract distributed brain regions that
exhibit similar :term:`BOLD` fluctuations over time. Decomposition
methods allow for generation of many independent maps simultaneously
without the need to provide a priori information (e.g. seeds or priors.)

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

from nilearn.decomposition import CanICA

canica = CanICA(
    n_components=20,
    memory="nilearn_cache",
    memory_level=2,
    verbose=10,
    mask_strategy="whole-brain-template",
    random_state=0,
    standardize="zscore_sample",
    n_jobs=2,
)
canica.fit(func_filenames)

# Retrieve the independent components in brain space. Directly
# accessible through attribute `components_img_`.
canica_components_img = canica.components_img_
# components_img is a Nifti Image object, and can be saved to a file with
# the following lines:
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_compare_decomposition"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
canica_components_img.to_filename(output_dir / "canica_resting_state.nii.gz")


# %%
# To visualize we plot the outline of all components on one figure
from nilearn.plotting import plot_prob_atlas

# Plot all ICA components together
plot_prob_atlas(canica_components_img, title="All ICA components")


# %%
# Finally, we plot the map for each :term:`ICA` component separately
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
# For more detailse see :footcite:t:`Mensch2016`.
#


# %%
# Create a dictionary learning estimator
from nilearn.decomposition import DictLearning

dict_learning = DictLearning(
    n_components=20,
    memory="nilearn_cache",
    memory_level=2,
    verbose=1,
    random_state=0,
    n_epochs=1,
    mask_strategy="whole-brain-template",
    standardize="zscore_sample",
    n_jobs=2,
)

print("[Example] Fitting dictionary learning model")
dict_learning.fit(func_filenames)
print("[Example] Saving results")
# Grab extracted components umasked back to Nifti image.
# Note: For older versions, less than 0.4.1. components_img_
# is not implemented. See Note section above for details.
dictlearning_components_img = dict_learning.components_img_
dictlearning_components_img.to_filename(
    output_dir / "dictionary_learning_resting_state.nii.gz"
)


# %%
# Visualize the results
#
# First plot all DictLearning components together
plot_prob_atlas(
    dictlearning_components_img, title="All DictLearning components"
)


# %%
# One plot of each component

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
# The fitted object `dict_learning` can be used
# to calculate the score per component
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
#     To see how to extract subject-level timeseries' from regions
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
