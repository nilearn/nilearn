"""
Predicted time series and residuals
===================================

Here we fit a First Level :term:`GLM` with the `minimize_memory`-argument
set to `False`.
By doing so, the `FirstLevelModel`-object stores the residuals, which we can
then inspect.
Also, the predicted time series can be extracted, which is useful to assess the
quality of the model fit.

"""

# %%
# Import modules
# --------------
import pandas as pd

from nilearn import image, masking
from nilearn.datasets import fetch_spm_auditory
from nilearn.plotting import plot_stat_map, show

# load fMRI data
subject_data = fetch_spm_auditory()
fmri_img = subject_data.func[0]

# Make an average
mean_img = image.mean_img(fmri_img)
mask = masking.compute_epi_mask(mean_img)

# Clean and smooth data
fmri_img = image.clean_img(fmri_img, standardize=None)
fmri_img = image.smooth_img(fmri_img, 5.0)

# load events
events = pd.read_csv(subject_data.events, sep="\t")


# %%
# Fit model
# ---------
# Note that `minimize_memory` is set to `False` so that `FirstLevelModel`
# stores the residuals.
# `signal_scaling` is set to False, so we keep the same scaling as the
# original data in `fmri_img`.
from nilearn.glm.first_level import FirstLevelModel

fmri_glm = FirstLevelModel(
    t_r=subject_data.t_r,
    drift_model="cosine",
    signal_scaling=False,
    mask_img=mask,
    minimize_memory=False,
    verbose=1,
)

fmri_glm = fmri_glm.fit(fmri_img, events)


# %%
# Calculate and plot contrast
# ---------------------------
z_map = fmri_glm.compute_contrast("listening")

threshold = 3.1
plot_stat_map(
    z_map,
    bg_img=mean_img,
    threshold=threshold,
    title=f"listening > rest (t-test; |Z|>{threshold})",
)

show()


# %%
# Extract the largest clusters
# ----------------------------
# We can extract the 6 largest clusters surviving our threshold.
# and get the x, y, and z coordinates of their peaks.
# We then extract the time series from a sphere around each coordinate.
#
from nilearn.maskers import NiftiSpheresMasker
from nilearn.reporting import get_clusters_table

table, _ = get_clusters_table(
    z_map,
    stat_threshold=threshold,
    cluster_threshold=20,
    return_label_maps=True,
)
table.set_index("Cluster ID", drop=True)
print(table)

# %%
# Find the coordinates of the 6 most significant clusters
coords = table.loc[range(1, 7), ["X", "Y", "Z"]].to_numpy()
print(coords)

masker = NiftiSpheresMasker(coords, verbose=1).fit(mean_img)

# let's have a look at the report to make sure the spheres are well placed.
report = masker.generate_report()
report

# %%
# Plot predicted and time series of the most significant clusters
# ---------------------------------------------------------------
# We can plot the predicted and observed time series together
# for a more direct comparison.
time_series_df, fig = fmri_glm.plot_predicted_signal_and_residuals(
    coords=coords[0:2], masker=masker, show=True
)


# %%
# Plot R-squared
# --------------
# Because we stored the residuals, we can plot the R-squared: the proportion of
# explained variance of the :term:`GLM` as a whole.
# Note that the R-squared is markedly
# lower deep down the brain, where there is more physiological noise and we are
# further away from the receive coils. However, R-Squared should be interpreted
# with a grain of salt. The R-squared value will necessarily increase with the
# addition of more factors (such as rest, active, drift, motion) into the GLM.
# Additionally, we are looking at the overall fit of the model, so we are
# unable to say whether a voxel/region has a large R-squared value because the
# voxel/region is responsive to the experiment (such as active or rest) or
# because the voxel/region fits the noise factors (such as drift or motion)
# that could be present in the :term:`GLM`.
# To isolate the influence of the experiment,
# we can use an F-test as shown in the next section.

plot_stat_map(
    fmri_glm.r_square[0],
    bg_img=mean_img,
    threshold=0.1,
    display_mode="z",
    cut_coords=7,
    cmap="inferno",
    title="R-squared",
    vmin=0,
    symmetric_cbar=False,
)


# %%
# Calculate and Plot F-test
# -------------------------
# The F-test tells you how well the :term:`GLM` fits effects of interest
# such as the active and rest conditions together.
# This is different from R-squared, which
# tells you how well the overall :term:`GLM` fits the data,
# including active, rest and all the other columns
# in the design matrix such as drift and motion.

# f-test for 'listening'
z_map_ftest = fmri_glm.compute_contrast(
    "listening", stat_type="F", output_type="z_score"
)

plot_stat_map(
    z_map_ftest,
    bg_img=mean_img,
    threshold=threshold,
    display_mode="z",
    cut_coords=7,
    cmap="inferno",
    title=f"listening > rest (F-test; Z>{threshold})",
    symmetric_cbar=False,
    vmin=0,
)

show()

# sphinx_gallery_dummy_images=2
