"""
Example of surface-based first-level analysis
=============================================

.. warning::

    This example is adapted from
    :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_localizer_surface_analysis.py`. # noqa
    to show how to use the new tentative API for surface images in nilearn.

    This functionality is provided
    by the :mod:`nilearn.experimental.surface` module.

    It is still incomplete and subject to change without a deprecation cycle.

    Please participate in the discussion on GitHub!

A full step-by-step example of fitting a :term:`GLM`
to experimental data sampled on the cortical surface
and visualizing the results.

More specifically:

1. A sequence of :term:`fMRI` volumes is loaded.
2. :term:`fMRI` data are projected onto a reference cortical surface
   (the FreeSurfer template, fsaverage).
3. A :term:`GLM` is applied to the dataset
   (effect/covariance, then contrast estimation).

The result of the analysis are statistical maps that are defined
on the brain mesh.
We display them using Nilearn capabilities.

The projection of :term:`fMRI` data onto a given brain :term:`mesh` requires
that both are initially defined in the same space.

* The functional data should be coregistered to the anatomy
  from which the mesh was obtained.

* Another possibility, used here, is to project
  the normalized :term:`fMRI` data to an :term:`MNI`-coregistered mesh,
  such as fsaverage.

The advantage of this second approach is that it makes it easy to run
second-level analyses on the surface.
On the other hand, it is obviously less accurate
than using a subject-tailored mesh.

"""

# %%
# Prepare data and analysis parameters
# ------------------------------------
#
# Prepare the timing parameters.
t_r = 2.4
slice_time_ref = 0.5

# %%
# Prepare the data.
# First, the volume-based :term:`fMRI` data.
from nilearn import datasets

data = datasets.fetch_localizer_first_level()
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


# %%
# We use the new :class:`nilearn.experimental.surface.SurfaceImage`
# to create an surface object instance
# that contains both the mesh
# (here we use the one from the fsaverage5 templates)
# and the BOLD data that we project on the surface.
from nilearn import surface
from nilearn.experimental.surface import SurfaceImage, load_fsaverage

fsaverage5 = load_fsaverage()
texture_left = surface.vol_to_surf(
    fmri_img, fsaverage5["pial"]["left_hemisphere"]
)
texture_right = surface.vol_to_surf(
    fmri_img, fsaverage5["pial"]["right_hemisphere"]
)
image = SurfaceImage(
    mesh={
        "lh": fsaverage5["pial"]["left_hemisphere"],
        "rh": fsaverage5["pial"]["right_hemisphere"],
    },
    data={
        "lh": texture_left.T,
        "rh": texture_right.T,
    },
)

# %%
# Perform first level analysis
# ----------------------------
#
# We can now simply run a GLM by directly passing
# our :class:`nilearn.experimental.surface.SurfaceImage` instance
# as input to FirstLevelModel.fit
#
# Here we use an :term:`HRF` model
# containing the Glover model and its time derivative
# The drift model is implicitly a cosine basis with a period cutoff at 128s.
from nilearn.glm.first_level import FirstLevelModel

glm = FirstLevelModel(
    t_r,
    slice_time_ref=slice_time_ref,
    hrf_model="glover + derivative",
).fit(image, events)

# %%
# Estimate contrasts
# ------------------
#
# Specify the contrasts.
#
# For practical purpose, we first generate an identity matrix whose size is
# the number of columns of the design matrix.
import numpy as np

design_matrix = glm.design_matrices_[0]
contrast_matrix = np.eye(design_matrix.shape[1])

# %%
# At first, we create basic contrasts.
basic_contrasts = {
    column: contrast_matrix[i]
    for i, column in enumerate(design_matrix.columns)
}

# %%
# Next, we add some intermediate contrasts and
# one :term:`contrast` adding all conditions with some auditory parts.
basic_contrasts["audio"] = (
    basic_contrasts["audio_left_hand_button_press"]
    + basic_contrasts["audio_right_hand_button_press"]
    + basic_contrasts["audio_computation"]
    + basic_contrasts["sentence_listening"]
)

# one contrast adding all conditions involving instructions reading
basic_contrasts["visual"] = (
    basic_contrasts["visual_left_hand_button_press"]
    + basic_contrasts["visual_right_hand_button_press"]
    + basic_contrasts["visual_computation"]
    + basic_contrasts["sentence_reading"]
)

# one contrast adding all conditions involving computation
basic_contrasts["computation"] = (
    basic_contrasts["visual_computation"]
    + basic_contrasts["audio_computation"]
)

# one contrast adding all conditions involving sentences
basic_contrasts["sentences"] = (
    basic_contrasts["sentence_listening"] + basic_contrasts["sentence_reading"]
)

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
    "left - right button press": (
        basic_contrasts["audio_left_hand_button_press"]
        - basic_contrasts["audio_right_hand_button_press"]
        + basic_contrasts["visual_left_hand_button_press"]
        - basic_contrasts["visual_right_hand_button_press"]
    ),
    "audio - visual": basic_contrasts["audio"] - basic_contrasts["visual"],
    "computation - sentences": (
        basic_contrasts["computation"] - basic_contrasts["sentences"]
    ),
}


# %%
# Let's estimate the contrasts by iterating over them.
from nilearn import plotting

fsaverage = datasets.fetch_surf_fsaverage()

for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print(
        f"  Contrast {index + 1:1} out of {len(contrasts)}: "
        f"{contrast_id}, right hemisphere"
    )
    # compute contrast-related statistics
    z_score = glm.compute_contrast(contrast_val, stat_type="t")

    # we plot it on the surface, on the inflated fsaverage mesh,
    # together with a suitable background to give an impression
    # of the cortex folding.
    plotting.plot_surf_stat_map(
        fsaverage.infl_right,
        z_score.data["rh"],
        hemi="right",
        title=contrast_id,
        colorbar=True,
        threshold=3.0,
        bg_map=fsaverage.sulc_right,
    )
    plotting.plot_surf_stat_map(
        fsaverage.infl_left,
        z_score.data["lh"],
        hemi="left",
        title=contrast_id,
        colorbar=True,
        threshold=3.0,
        bg_map=fsaverage.sulc_right,
    )

plotting.show()
