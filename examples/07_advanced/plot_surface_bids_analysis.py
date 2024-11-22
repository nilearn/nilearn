"""
Surface-based dataset first and second level analysis of a dataset
==================================================================

Full step-by-step example of fitting a :term:`GLM`
(first and second level analysis) in a 10-subjects dataset
and visualizing the results.

More specifically:

#. Download an :term:`fMRI` :term:`BIDS` dataset
   with two language conditions to contrast.

#. Project the data to a standard mesh, fsaverage5,
   also known as the Freesurfer template :term:`mesh`
   downsampled to about 10k nodes per hemisphere.

#. Run the first level model objects.

#. Fit a second level model on the fitted first level models.

Notice that in this case the preprocessed :term:`bold<BOLD>` images
were already normalized to the same :term:`MNI` space.
"""

# %%
# Fetch example :term:`BIDS` dataset
# ----------------------------------
# We download a simplified :term:`BIDS` dataset
# made available for illustrative purposes.
# It contains only the necessary information
# to run a statistical analysis using Nilearn.
# The raw data subject folders only contain bold.json and events.tsv files,
# while the derivatives folder includes the preprocessed files preproc.nii
# and the confounds.tsv files.
from nilearn.datasets import fetch_language_localizer_demo_dataset

data = fetch_language_localizer_demo_dataset(legacy_output=False)

# %%
# Here is the location of the dataset on disk.
data.data_dir

# %%
# Obtain automatically FirstLevelModel objects and fit arguments
# --------------------------------------------------------------
# From the dataset directory we automatically obtain
# the FirstLevelModel objects
# with their subject_id filled from the :term:`BIDS` dataset.
# Along, we also obtain:
#
#   - a list with the Nifti image associated with each run
#
#   - a list of events read from events.tsv in the the :term:`BIDS` dataset
#
#   - a list of confounder motion regressors
#     since in this case a confounds.tsv file is available
#     in the :term:`BIDS` dataset.
#
# To get the first level models we only have to specify the dataset directory
# and the ``task_label`` as specified in the file names.
#
# .. note::
#
#       We are only using a subset of participants from the dataset
#       to lower the run time of the example.
#
from nilearn.glm.first_level import first_level_from_bids

models, run_imgs, events, confounds = first_level_from_bids(
    dataset_path=data.data_dir,
    task_label="languagelocalizer",
    img_filters=[("desc", "preproc")],
    sub_labels=["01", "02", "03", "04", "05"],  # comment to run all subjects
    hrf_model="glover + derivative",
    n_jobs=-1,
)

# %%
# Project :term:`fMRI` data to the surface and compute GLM and contrasts
#
# The projection function simply takes the :term:`fMRI` data and the mesh.
# Note that those correspond spatially, as they are both in :term:`MNI` space.
#
# .. warning::
#
#    Note that here we pass ALL the confounds when we fit the model.
#    In this case we can do this because our regressors only include
#    the motion realignment parameters.
#    For most preprocessed BIDS dataset,
#    you would have to carefully choose which confounds to include.
#
from pathlib import Path

from nilearn.datasets import load_fsaverage
from nilearn.surface import SurfaceImage

fsaverage5 = load_fsaverage()

# Empty lists in which we are going to store activation values.
z_scores = []
z_scores_left = []
z_scores_right = []
for first_level_glm, fmri_img, confound, event in zip(
    models, run_imgs, confounds, events
):
    print(f"Running GLM on {Path(fmri_img[0]).relative_to(data.data_dir)}")

    image = SurfaceImage.from_volume(
        mesh=fsaverage5["pial"],
        volume_img=fmri_img[0],
    )

    # Fit GLM.
    # Pass events and all confounds
    first_level_glm.fit(run_imgs=image, events=event[0], confounds=confound[0])

    # Compute contrast between 'language' and 'string' events
    z_scores.append(
        first_level_glm.compute_contrast("language-string", stat_type="t")
    )
    z_scores_left.append(z_scores[-1].data.parts["left"])
    z_scores_right.append(z_scores[-1].data.parts["right"])


# %%
# Group study
# -----------
#
# Individual activation maps have been accumulated
# in the ``z_score_left`` and ``z_scores_right`` lists respectively.
# We can now use them in a group study (one-sample study).
#
# Prepare figure for concurrent plot of individual maps
# compute population-level maps for left and right hemisphere
# We directly do that on the value arrays.
import numpy as np
from scipy.stats import norm, ttest_1samp

_, pval_left = ttest_1samp(np.array(z_scores_left), 0)
_, pval_right = ttest_1samp(np.array(z_scores_right), 0)

# %%
# What we have so far are p-values: we convert them to z-values for plotting.
z_val_left = norm.isf(pval_left)
z_val_right = norm.isf(pval_right)

# %%
# Plot the resulting maps, at first on the left hemisphere.
from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf_stat_map, show

fsaverage_data = load_fsaverage_data(data_type="sulcal")

for hemi, stat_map in zip(["left", "right"], [z_val_left, z_val_right]):
    plot_surf_stat_map(
        surf_mesh=fsaverage5["inflated"],
        stat_map=stat_map,
        hemi=hemi,
        title=f"(language-string), {hemi} hemisphere ; scipy",
        colorbar=True,
        cmap="bwr",
        threshold=1.96,
        bg_map=fsaverage_data,
    )


# %%
# Use SecondLevel

import pandas as pd

from nilearn.glm.second_level import SecondLevelModel

second_level_glm = SecondLevelModel()
design_matrix = pd.DataFrame([1] * len(z_scores), columns=["intercept"])
second_level_glm.fit(second_level_input=z_scores, design_matrix=design_matrix)
results = second_level_glm.compute_contrast("intercept", output_type="z_score")

for hemi in ["left", "right"]:
    plot_surf_stat_map(
        surf_mesh=fsaverage5["inflated"],
        stat_map=results,
        hemi=hemi,
        title=f"(language-string), {hemi} hemisphere",
        colorbar=True,
        cmap="bwr",
        threshold=1.96,
        bg_map=fsaverage_data,
    )

show()
