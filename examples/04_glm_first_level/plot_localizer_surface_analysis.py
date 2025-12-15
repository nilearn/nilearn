"""
Example of surface-based first-level analysis
=============================================

A full step-by-step example of fitting a :term:`GLM`
to experimental data sampled on the cortical surface
and visualizing the results.

More specifically:

1. A sequence of :term:`fMRI` volumes is loaded.
2. :term:`fMRI` data are projected onto a reference cortical surface
   (the FreeSurfer template, fsaverage).
3. A :term:`GLM` is applied to the dataset
   (effect/covariance, then contrast estimation).
4. Inspect GLM reports and save the results to disk.

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

# %%
# Fetch the data.
from nilearn.datasets import fetch_localizer_first_level

data = fetch_localizer_first_level()
t_r = data.t_r
slice_time_ref = data.slice_time_ref

# %%
# Project the :term:`fMRI` image to the surface
# ---------------------------------------------
#
# For this we need to get a :term:`mesh`
# representing the geometry of the surface.
# We could use an individual :term:`mesh`,
# but we first resort to a standard :term:`mesh`,
# the so-called fsaverage5 template from the FreeSurfer software.
#
# We use the :class:`~nilearn.surface.SurfaceImage`
# to create an surface object instance
# that contains both the mesh
# (here we use the one from the fsaverage5 templates)
# and the BOLD data that we project on the surface.

from nilearn.datasets import load_fsaverage
from nilearn.surface import SurfaceImage

fsaverage5 = load_fsaverage()
surface_image = SurfaceImage.from_volume(
    mesh=fsaverage5["pial"],
    volume_img=data.epi_img,
)

# %%
# Perform first level analysis
# ----------------------------
#
# We can now simply run a GLM by directly passing
# our :class:`~nilearn.surface.SurfaceImage` instance
# as input to FirstLevelModel.fit
#
# Here we use an :term:`HRF` model
# containing the Glover model and its time derivative
# The drift model is implicitly a cosine basis with a period cutoff at 128s.
from nilearn.glm.first_level import FirstLevelModel

glm = FirstLevelModel(
    t_r=t_r,
    slice_time_ref=slice_time_ref,
    hrf_model="glover + derivative",
    minimize_memory=False,
    verbose=1,
).fit(run_imgs=surface_image, events=data.events)

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

# %%
# one contrast adding all conditions involving instructions reading
basic_contrasts["visual"] = (
    basic_contrasts["visual_left_hand_button_press"]
    + basic_contrasts["visual_right_hand_button_press"]
    + basic_contrasts["visual_computation"]
    + basic_contrasts["sentence_reading"]
)

# %%
# one contrast adding all conditions involving computation
basic_contrasts["computation"] = (
    basic_contrasts["visual_computation"]
    + basic_contrasts["audio_computation"]
)

# %%
# one contrast adding all conditions involving sentences
basic_contrasts["sentences"] = (
    basic_contrasts["sentence_listening"] + basic_contrasts["sentence_reading"]
)

# %%
# Finally, we create a dictionary of more relevant contrasts
#
# * ``'left - right button press'``: probes motor activity
#   in left versus right button presses.
# * ``'audio - visual'``: probes the difference of activity between listening
#   to some content or reading the same type of content
#   (instructions, stories).
# * ``'computation - sentences'``: looks at the activity
#   when performing a mental computation task  versus simply reading sentences.
#
# Of course, we could define other contrasts,
# but we keep only 3 for simplicity.

contrasts = {
    "(left - right) button press": (
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
# Let's estimate the t-contrasts,
# and extract a table of clusters
# that survive our thresholds.
#
# We use the same threshold (uncorrected p < 0.001)
# for all contrasts.
#
# We plot each contrast map on the inflated fsaverage mesh,
# together with a suitable background to give an impression
# of the cortex folding.
#

from scipy.stats import norm

from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf_stat_map, show
from nilearn.reporting import get_clusters_table

p_val = 0.001
threshold = norm.isf(p_val)
cluster_threshold = 20
two_sided = True

fsaverage_data = load_fsaverage_data(data_type="sulcal")

results = {}
for contrast_id, contrast_val in contrasts.items():
    results[contrast_id] = glm.compute_contrast(contrast_val, stat_type="t")

    table = get_clusters_table(
        results[contrast_id],
        stat_threshold=threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
    )
    print(f"\n{contrast_id=}")
    print(table)

for contrast_id, z_score in results.items():
    hemi = "left"
    if contrast_id == "(left - right) button press":
        hemi = "both"

    plot_surf_stat_map(
        surf_mesh=fsaverage5["inflated"],
        stat_map=z_score,
        hemi=hemi,
        title=contrast_id,
        threshold=threshold,
        bg_map=fsaverage_data,
    )

show()

# %%
# Cluster-level inference
# -----------------------
#
# We can also perform cluster-level inference
# (aka "All resolution Inference") for a given contrast.
# This gives a high-probability lower bound
# on the proportion of true discoveries in each cluster
#
from nilearn.glm import cluster_level_inference
from nilearn.surface.surface import get_data as get_surf_data

proportion_true_discoveries_img = cluster_level_inference(
    results["audio - visual"], threshold=[2.5, 3.5, 4.5], alpha=0.05
)

data = get_surf_data(proportion_true_discoveries_img)
unique_vals = np.unique(data.ravel())
print(unique_vals)

plot_surf_stat_map(
    surf_mesh=fsaverage5["inflated"],
    stat_map=proportion_true_discoveries_img,
    hemi="left",
    cmap="inferno",
    title="audio - visual, proportion true positives",
    bg_map=fsaverage_data,
    avg_method="max",
)

show()

# %%
# Generate a report for the GLM
# -----------------------------
#
report = glm.generate_report(
    contrasts,
    threshold=threshold,
    bg_img=load_fsaverage_data(data_type="sulcal", mesh_type="inflated"),
    height_control=None,
    cluster_threshold=cluster_threshold,
    two_sided=two_sided,
)

# %%
#
# .. include:: ../../../examples/report_note.rst
#
report
