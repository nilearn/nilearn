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
# Subject level models
# --------------------
# From the dataset directory we automatically obtain
# the FirstLevelModel objects
# with their subject_id filled from the :term:`BIDS` dataset.
# Along, we also obtain:
#
# - a list with the Nifti image associated with each run
#
# - a list of events read from events.tsv in the :term:`BIDS` dataset
#
# - a list of confounder motion regressors
#   since in this case a confounds.tsv file is available
#   in the :term:`BIDS` dataset.
#
# To get the first level models we only have to specify the dataset directory
# and the ``task_label`` as specified in the file names.
#
#
from nilearn.glm.first_level import first_level_from_bids

models, run_imgs, events, confounds = first_level_from_bids(
    dataset_path=data.data_dir,
    task_label="languagelocalizer",
    space_label="",
    img_filters=[("desc", "preproc")],
    n_jobs=2,
)

# %%
# Project :term:`fMRI` data to the surface, fit the GLM and compute contrasts
#
# The projection function simply takes the :term:`fMRI` data and the mesh.
# Note that those correspond spatially, as they are both in same space.
#
# .. warning::
#
#    Note that here we pass ALL the confounds when we fit the model.
#    In this case we can do this because our regressors only include
#    the motion realignment parameters.
#    For most preprocessed BIDS dataset,
#    you would have to carefully choose which confounds to include.
#
#    When working with a typical BIDS derivative dataset
#    generated by fmriprep,
#    the :obj:`~nilearn.glm.first_level.first_level_from_bids` function
#    allows you to indirectly pass arguments to
#    :obj:`~nilearn.interfaces.fmriprep.load_confounds`,
#    so you can selectively load specific subsets of confounds
#    to implement certain denoising strategies.
#
from pathlib import Path

from nilearn.datasets import load_fsaverage
from nilearn.surface import SurfaceImage

fsaverage5 = load_fsaverage()

# Empty lists in which we are going to store activation values.
z_scores = []
z_scores_left = []
z_scores_right = []
for i, (first_level_glm, fmri_img, confound, event) in enumerate(
    zip(models, run_imgs, confounds, events)
):
    print(f"Running GLM on {Path(fmri_img[0]).relative_to(data.data_dir)}")

    image = SurfaceImage.from_volume(
        mesh=fsaverage5["pial"],
        volume_img=fmri_img[0],
    )

    # Fit GLM.
    # Pass events and all confounds
    first_level_glm.fit(
        run_imgs=image,
        events=event[0],
        confounds=confound[0],
    )

    # Compute contrast between 'language' and 'string' events
    z_scores.append(
        first_level_glm.compute_contrast(
            "language-string", stat_type="t", output_type="z_score"
        )
    )

    # Let's only generate a report for the first subject
    if i == 1:
        report_flm = first_level_glm.generate_report(
            contrasts="language-string", threshold=1.96, alpha=0.001
        )

# View the GLM report of the first subject
report_flm

# %%
# Group level model
# -----------------
#
# Individual activation maps have been accumulated in the ``z_score``.
# We can now use them in a one-sample t-test at the group level model
# by passing them as input
# to :class:`~nilearn.glm.second_level.SecondLevelModel`.
#
import pandas as pd

from nilearn.glm.second_level import SecondLevelModel

second_level_glm = SecondLevelModel()
design_matrix = pd.DataFrame([1] * len(z_scores), columns=["intercept"])
second_level_glm.fit(second_level_input=z_scores, design_matrix=design_matrix)

results = second_level_glm.compute_contrast("intercept", output_type="z_score")

report_slm = second_level_glm.generate_report(
    contrasts="intercept", threshold=1.96, alpha=0.001
)

# View the GLM report at the group level
report_slm


# %%
# Visualization
# -------------
# We can now plot
# the computed group-level maps for left and right hemisphere
from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf_stat_map, show

fsaverage_data = load_fsaverage_data(data_type="sulcal")

for hemi in ["left", "right"]:
    plot_surf_stat_map(
        surf_mesh=fsaverage5["inflated"],
        stat_map=results,
        hemi=hemi,
        title=f"(language-string), {hemi} hemisphere",
        colorbar=True,
        threshold=1.96,
        bg_map=fsaverage_data,
    )

show()
