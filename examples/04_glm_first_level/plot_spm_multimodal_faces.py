"""
Single-subject data (two runs) in native space
==============================================

The example shows the analysis of an :term:`SPM` dataset,
with two conditions: viewing a face image or a scrambled face image.

This example takes a lot of time because the input are lists of 3D images
sampled in different positions (encoded by different affine functions).

.. seealso::

    For more information
    see the :ref:`dataset description <spm_multimodal_dataset>`.
"""

# %%
# Fetch and inspect the data
# --------------------------
# Fetch the :term:`SPM` multimodal_faces data.
from nilearn.datasets import fetch_spm_multimodal_fmri

subject_data = fetch_spm_multimodal_fmri()

# %%
# Let's inspect one of the event files before using them.
import pandas as pd

events = [subject_data.events1, subject_data.events2]

events_dataframe = pd.read_csv(events[0], sep="\t")
events_dataframe["trial_type"].value_counts()

# %%
# We can confirm there are only 2 conditions in the dataset.
#
from nilearn.plotting import plot_event, show

plot_event(events)

show()

# %%
# Resample the images:
# this is achieved by the ``concat_imgs`` function of Nilearn.
import warnings

from nilearn.image import concat_imgs, mean_img, resample_img

# Avoid getting too many warnings due to resampling
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fmri_img = [
        concat_imgs(subject_data.func1, auto_resample=True),
        concat_imgs(subject_data.func2, auto_resample=True),
    ]
affine, shape = fmri_img[0].affine, fmri_img[0].shape
print("Resampling the second image (this takes time)...")
fmri_img[1] = resample_img(fmri_img[1], affine, shape[:3], copy_header=True)

# %%
# Let's create mean image for display purposes.
mean_image = mean_img(fmri_img, copy_header=True)

# %%
# Fit the model
# -------------
# Fit the :term:`GLM` for the 2 runs
# by specifying a FirstLevelModel and then fitting it.

# Sample at the beginning of each acquisition.
slice_time_ref = 0.0
# We use a discrete cosine transform to model signal drifts.
drift_model = "cosine"
# The cutoff for the drift model is 0.01 Hz.
high_pass = 0.01
# The hemodynamic response function
hrf_model = "spm + derivative"

from nilearn.glm.first_level import FirstLevelModel

print("Fitting a GLM")
fmri_glm = FirstLevelModel(
    smoothing_fwhm=None,
    t_r=subject_data.t_r,
    hrf_model=hrf_model,
    drift_model=drift_model,
    high_pass=high_pass,
)


fmri_glm = fmri_glm.fit(fmri_img, events=events)

# %%
# View the results
# ----------------
# Now we can compute contrast-related statistical maps (in z-scale),
# and plot them.
from nilearn.plotting import plot_stat_map

print("Computing contrasts")

# %%
# We actually want more interesting contrasts.
# The simplest contrast just makes the difference
# between the two main conditions.
# We define the two opposite versions to run one-tailed t-tests.
#

contrasts = ["faces - scrambled", "scrambled - faces"]

# %%
# Let's store common parameters for all plots.
#
# We plot the contrasts values overlaid on the mean fMRI image
# and we will use the z-score values as transparency,
# with any voxel with | Z-score | > 3 being fully opaque
# and any voxel with 0 < | Z-score | < 1.96 being partly transparent.
plot_param = {
    "vmin": 0,
    "display_mode": "z",
    "cut_coords": 3,
    "black_bg": True,
    "bg_img": mean_image,
    "cmap": "inferno",
    "transparency_range": [0, 3],
}

# Iterate on contrasts to compute and plot them.
for contrast_id in contrasts:
    print(f"\tcontrast id: {contrast_id}")

    results = fmri_glm.compute_contrast(contrast_id, output_type="all")

    plot_stat_map(
        results["stat"],
        title=contrast_id,
        transparency=results["z_score"],
        **plot_param,
    )

# %%
# We also define the effects of interest contrast,
# a 2-dimensional contrasts spanning the two conditions.
#
import numpy as np

contrasts = np.eye(2)

results = fmri_glm.compute_contrast(contrasts, output_type="all")

plot_stat_map(
    results["stat"],
    title="effects of interest",
    transparency=results["z_score"],
    **plot_param,
)

show()

# %%
# Based on the resulting maps we observe
# that the analysis results in wide activity
# for the 'effects of interest' contrast,
# showing the implications of large portions of the visual cortex
# in the conditions.
# By contrast,
# the differential effect between "faces" and "scrambled" involves sparser,
# more anterior and lateral regions.
# It also displays some responses in the frontal lobe.

# sphinx_gallery_dummy_images=4
