"""
Single-subject data (two runs) in native space
==============================================

The example shows the analysis of an :term:`SPM` dataset
studying face perception.
The analysis is performed in native space
and have not been resampled to a common space.

The experimental paradigm is simple, with two conditions;
viewing a face image or a scrambled face image,
supposedly with the same low-level statistical properties,
to find face-specific responses.

For details on the data, please see :footcite:t:`Henson2003`.

This example takes a lot of time because the input are lists of 3D images
sampled in different positions (encoded by different affine functions).
"""

# %%
# Fetch the :term:`SPM` multimodal_faces data.
from nilearn.datasets import fetch_spm_multimodal_fmri

subject_data = fetch_spm_multimodal_fmri()

# %%
# Specify timing and design matrix parameters.

# repetition time, in seconds
t_r = 2.0
# Sample at the beginning of each acquisition.
slice_time_ref = 0.0
# We use a discrete cosine transform to model signal drifts.
drift_model = "Cosine"
# The cutoff for the drift model is 0.01 Hz.
high_pass = 0.01
# The hemodynamic response function
hrf_model = "spm + derivative"

# %%
# Resample the images.
#
# This is achieved by the concat_imgs function of Nilearn.
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
fmri_img[1] = resample_img(
    fmri_img[1], affine, shape[:3], copy_header=True, force_resample=True
)

# %%
# Let's create mean image for display purposes.
mean_image = mean_img(fmri_img, copy_header=True)

# %%
# Make the design matrices.
import numpy as np
import pandas as pd

from nilearn.glm.first_level import make_first_level_design_matrix

design_matrices = []

# %%
# Loop over the two runs.
for idx, img in enumerate(fmri_img, start=1):
    # Build experimental paradigm
    n_scans = img.shape[-1]
    events = pd.read_table(subject_data[f"events{idx}"])
    # Define the sampling times for the design matrix
    frame_times = np.arange(n_scans) * t_r
    # Build design matrix with the previously defined parameters
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
    )

    # put the design matrices in a list
    design_matrices.append(design_matrix)

# %%
# Fit the :term:`GLM` for the 2 runs
# by specifying a FirstLevelModel and then fitting it.
from nilearn.glm.first_level import FirstLevelModel

print("Fitting a GLM")
fmri_glm = FirstLevelModel(smoothing_fwhm=6)
fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)

# %%
# Now we can compute contrast-related statistical maps (in z-scale),
# and plot them.
from nilearn.plotting import plot_stat_map, show

print("Computing contrasts")

# %%
# We can specify basic contrasts
# (to get :term:`beta<Parameter Estimate>` maps).
# We start by specifying canonical :term:`contrast`
# that isolate design matrix columns.
contrast_matrix = np.eye(design_matrix.shape[1])
basic_contrasts = {
    column: contrast_matrix[i]
    for i, column in enumerate(design_matrix.columns)
}

# %%
# We actually want more interesting contrasts.
# The simplest contrast just makes the difference
# between the two main conditions.
# We define the two opposite versions to run one-tailed t-tests.
#

contrasts = ["faces-scrambled", "scrambled-faces"]


# %%
# Let's store common parameters for all plots
#
# We plot the contrasts values overlaid on the mean fMRI image
# and we will use the z-score values as transparency,
# with any voxel with | Z-score | > 3 being fully opaque
# and any voxel with | Z-score | < 1.96 being fully transparent.
plot_param = {
    "threshold": 0,
    "vmin": 0,
    "display_mode": "z",
    "cut_coords": [-40, -25, -6],
    "black_bg": True,
    "bg_img": mean_image,
    "cmap": "inferno",
    "transparency_range": [1.96, 3],
}

# Iterate on contrasts to compute an plot them.
for contrast_id in contrasts:
    print(f"\tcontrast id: {contrast_id}")

    results = fmri_glm.compute_contrast(contrast_id, output_type="all")

    plot_stat_map(
        results["stat"],
        title=contrast_id,
        transparency=results["z_score"],
        vmax=6,
        **plot_param,
    )

# %%
# We also define the effects of interest contrast,
# a 2-dimensional contrasts spanning the two conditions.
#

contrasts = {
    "effects_of_interest": np.vstack(
        (basic_contrasts["faces"], basic_contrasts["scrambled"])
    ),
}

for contrast_id, contrast_val in contrasts.items():
    print(f"\tcontrast id: {contrast_id}")

    results = fmri_glm.compute_contrast(contrast_val, output_type="all")

    plot_stat_map(
        results["stat"],
        title=contrast_id,
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

# %%
# References
# ----------
#
# .. footbibliography::


# sphinx_gallery_dummy_images=3
