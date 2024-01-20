"""
Example of surface-based first-level analysis
=============================================

A full step-by-step example of fitting a :term:`GLM` to experimental
data sampled on the cortical surface and visualizing the results.

More specifically:

1. A sequence of :term:`fMRI` volumes is loaded.
2. :term:`fMRI` data are projected onto a reference cortical surface
   (the FreeSurfer template, fsaverage).
3. A design matrix describing all the effects related to the data is computed.
4. A :term:`GLM` is applied to the dataset
   (effect/covariance, then contrast estimation).

The result of the analysis are statistical maps that are defined on the brain
mesh. We display them using Nilearn capabilities.

The projection of :term:`fMRI` data onto a given brain :term:`mesh` requires
that both are initially defined in the same space.

* The functional data should be coregistered to the anatomy from which the mesh
  was obtained.

* Another possibility, used here, is to project
  the normalized :term:`fMRI` data to an :term:`MNI`-coregistered mesh,
  such as fsaverage.

The advantage of this second approach is that it makes it easy to run
second-level analyses on the surface. On the other hand, it is obviously less
accurate than using a subject-tailored mesh.

"""

# %%
# Prepare data and analysis parameters
# ------------------------------------
# Prepare the timing parameters.
t_r = 2.4
slice_time_ref = 0.5

# %%
# Prepare the data.
# First, the volume-based :term:`fMRI` data.
from nilearn.datasets import fetch_localizer_first_level

data = fetch_localizer_first_level()
fmri_img = data.epi_img

# %%
# Second, the experimental paradigm.
import pandas as pd

events_file = data.events
events = pd.read_table(events_file)

# %%
# Project the :term:`fMRI` image to the surface
# ---------------------------------------------
#
# For this we need to get a :term:`mesh`
# representing the geometry of the surface.
# We could use an individual :term:`mesh`,
# but we first resort to a standard :term:`mesh`,
# the so-called fsaverage5 template from the FreeSurfer software.
import nilearn

fsaverage = nilearn.datasets.fetch_surf_fsaverage()

# %%
# The projection function simply takes the :term:`fMRI` data and the mesh.
# Note that those correspond spatially, are they are both in :term:`MNI` space.
from nilearn import surface

texture = surface.vol_to_surf(fmri_img, fsaverage.pial_right)

# %%
# Perform first level analysis
# ----------------------------
#
# This involves computing the design matrix and fitting the model.
# We start by specifying the timing of :term:`fMRI` frames.
import numpy as np

n_scans = texture.shape[1]
frame_times = t_r * (np.arange(n_scans) + .5)

# %%
# Create the design matrix.
#
# We specify an :term:`HRF` model
# containing the Glover model and its time derivative
# The drift model is implicitly a cosine basis with a period cutoff at 128s.
from nilearn.glm.first_level import make_first_level_design_matrix

design_matrix = make_first_level_design_matrix(frame_times,
                                               events=events,
                                               hrf_model='glover + derivative'
                                               )

# %%
# Setup and fit GLM.
#
# Note that the output consists in 2 variables: `labels` and `fit`.
# `labels` tags voxels according to noise autocorrelation.
# `estimates` contains the parameter estimates.
# We keep them for later :term:`contrast` computation.
from nilearn.glm.first_level import run_glm

labels, estimates = run_glm(texture.T, design_matrix.values)

# %%
# Estimate contrasts
# ------------------
# Specify the contrasts.
#
# For practical purpose, we first generate an identity matrix whose size is
# the number of columns of the design matrix.
contrast_matrix = np.eye(design_matrix.shape[1])

# %%
# At first, we create basic contrasts.
basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])

# %%
# Next, we add some intermediate contrasts and
# one :term:`contrast` adding all conditions with some auditory parts.
basic_contrasts['audio'] = (
    basic_contrasts['audio_left_hand_button_press']
    + basic_contrasts['audio_right_hand_button_press']
    + basic_contrasts['audio_computation']
    + basic_contrasts['sentence_listening'])

# one contrast adding all conditions involving instructions reading
basic_contrasts['visual'] = (
    basic_contrasts['visual_left_hand_button_press']
    + basic_contrasts['visual_right_hand_button_press']
    + basic_contrasts['visual_computation']
    + basic_contrasts['sentence_reading'])

# one contrast adding all conditions involving computation
basic_contrasts['computation'] = (basic_contrasts['visual_computation']
                                  + basic_contrasts['audio_computation'])

# one contrast adding all conditions involving sentences
basic_contrasts['sentences'] = (basic_contrasts['sentence_listening']
                                + basic_contrasts['sentence_reading'])

# %%
# Finally, we create a dictionary of more relevant contrasts
#
# * 'left - right button press': probes motor activity
#   in left versus right button presses.
# * 'audio - visual': probes the difference of activity between listening
#   to some content or reading the same type of content
#   (instructions, stories).
# * 'computation - sentences': looks at the activity
#   when performing a mental computation task  versus simply reading sentences.
#
# Of course, we could define other contrasts,
# but we keep only 3 for simplicity.

contrasts = {
    'left - right button press': (
        basic_contrasts['audio_left_hand_button_press']
        - basic_contrasts['audio_right_hand_button_press']
        + basic_contrasts['visual_left_hand_button_press']
        - basic_contrasts['visual_right_hand_button_press']
    ),
    'audio - visual': basic_contrasts['audio'] - basic_contrasts['visual'],
    'computation - sentences': (
        basic_contrasts['computation']
        - basic_contrasts['sentences']
    )
}

# %%
# Let's estimate the contrasts by iterating over them.
from nilearn import plotting
from nilearn.glm.contrasts import compute_contrast

for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print(f"  Contrast {index + 1:1} out of {len(contrasts)}: "
          f"{contrast_id}, right hemisphere")
    # compute contrast-related statistics
    contrast = compute_contrast(labels, estimates, contrast_val,
                                stat_type='t')
    # we present the Z-transform of the t map
    z_score = contrast.z_score()
    # we plot it on the surface, on the inflated fsaverage mesh,
    # together with a suitable background to give an impression
    # of the cortex folding.
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, z_score, hemi='right',
        title=contrast_id, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_right)

# %%
# Analysing the left hemisphere
# -----------------------------
#
# Note that re-creating the above analysis for the left hemisphere requires
# little additional code!

# %%
# We project the :term:`fMRI` data to the mesh.
texture = surface.vol_to_surf(fmri_img, fsaverage.pial_left)

# %%
# Then we estimate the General Linear Model.
labels, estimates = run_glm(texture.T, design_matrix.values)

# %%
# Finally, we create contrast-specific maps and plot them.
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print(f"  Contrast {index + 1:1} out of {len(contrasts)}: "
          f"{contrast_id}, left hemisphere")
    # compute contrasts
    contrast = compute_contrast(labels, estimates, contrast_val,
                                stat_type='t')
    z_score = contrast.z_score()
    # plot the result
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, z_score, hemi='left',
        title=contrast_id, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_left)

plotting.show()
