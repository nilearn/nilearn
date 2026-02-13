"""
Default Mode Network extraction of ADHD dataset
===============================================

This example shows a full step-by-step workflow of fitting a :term:`GLM`
to signal extracted from a seed on the Posterior Cingulate Cortex
and saving the results.
More precisely, this example shows how to use a signal extracted from a
seed region as the regressor in a :term:`GLM` to determine the correlation
of each region in the dataset with the seed region.

More specifically:

1. A sequence of :term:`fMRI` volumes are loaded.
2. A design matrix with the Posterior Cingulate Cortex seed is defined.
3. A :term:`GLM` is applied to the dataset (effect/covariance,
   then contrast estimation).
4. The Default Mode Network is displayed.

"""

# %%
import numpy as np

from nilearn import plotting
from nilearn.datasets import fetch_adhd
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)
from nilearn.maskers import NiftiSpheresMasker

# %%
# Prepare data and analysis parameters
# ------------------------------------
# Prepare the data.
adhd_dataset = fetch_adhd(n_subjects=1)

# Prepare seed
pcc_coords = (0, -53, 26)

# %%
# Extract the seed region's time course
# -------------------------------------
# Extract the time course of the seed region.
seed_masker = NiftiSpheresMasker(
    [pcc_coords],
    radius=10,
    detrend=True,
    standardize="zscore_sample",
    low_pass=0.1,
    high_pass=0.01,
    t_r=adhd_dataset.t_r,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])

n_scans = seed_time_series.shape[0]
frametimes = np.linspace(0, (n_scans - 1) * adhd_dataset.t_r, n_scans)

# %%
# Plot the time course of the seed region.
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 3))
ax = fig.add_subplot(111)
ax.plot(frametimes, seed_time_series, linewidth=2, label="seed region")
ax.legend(loc=2)
ax.set_title("Time course of the seed region")
plt.show()

# %%
# Estimate contrasts
# ------------------
# Specify the contrasts.
design_matrix = make_first_level_design_matrix(
    frametimes,
    hrf_model="spm",
    add_regs=seed_time_series,
    add_reg_names=["pcc_seed"],
)
dmn_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
contrasts = {"seed_based_glm": dmn_contrast}

# %%
# Perform first level analysis
# ----------------------------
# Setup and fit GLM.
first_level_model = FirstLevelModel(verbose=1)
first_level_model = first_level_model.fit(
    run_imgs=adhd_dataset.func[0], design_matrices=design_matrix
)

# %%
# Estimate the contrast.
print("Contrast seed_based_glm computed.")
z_map = first_level_model.compute_contrast(
    contrasts["seed_based_glm"], output_type="z_score"
)

# %%
# Saving snapshots of the contrasts
from pathlib import Path

display = plotting.plot_stat_map(
    z_map, threshold=3.0, title="Seed based GLM", cut_coords=pcc_coords
)
display.add_markers(
    marker_coords=[pcc_coords], marker_color="g", marker_size=300
)

output_dir = Path.cwd() / "results" / "plot_adhd_dmn"
output_dir.mkdir(exist_ok=True, parents=True)
filename = "dmn_z_map.png"
display.savefig(output_dir / filename)
print(f"Save z-map in '{filename}'.")

# %%
# Generating a report
# -------------------
# It can be useful to quickly generate a
# portable, ready-to-view report with most of the pertinent information.
# This is easy to do if you have a fitted model and the list of contrasts,
# which we do here.

report = first_level_model.generate_report(
    contrasts=contrasts,
    title="ADHD DMN Report",
    cluster_threshold=15,
    min_distance=8.0,
    plot_type="glass",
)

# %%
#
# .. include:: ../../../examples/report_note.rst
#
report
