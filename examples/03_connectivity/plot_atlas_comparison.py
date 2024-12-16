"""
Comparing connectomes on different reference atlases
====================================================

This examples shows how to turn a :term:`parcellation` into connectome for
visualization. This requires choosing centers for each parcel
or network, via :func:`~nilearn.plotting.find_parcellation_cut_coords` for
:term:`parcellation` based on labels and
:func:`~nilearn.plotting.find_probabilistic_atlas_cut_coords` for
:term:`parcellation` based on probabilistic values.

In the intermediary steps, we make use of
:class:`~nilearn.maskers.MultiNiftiLabelsMasker` and
:class:`~nilearn.maskers.MultiNiftiMapsMasker`
to extract time series from nifti
objects from multiple subjects using different :term:`parcellation` atlases.

The time series of all subjects of the brain development dataset are
concatenated and given directly to
:class:`~nilearn.connectome.ConnectivityMeasure` for computing parcel-wise
correlation matrices for each atlas across all subjects.

Mean correlation matrix is displayed on glass brain on extracted coordinates.

# author: Amadeus Kanaan

.. include:: ../../../examples/masker_note.rst

"""

# %%
# Load atlases
# ------------
from nilearn import datasets

yeo = datasets.fetch_atlas_yeo_2011()
print(
    "Yeo atlas nifti image (3D) with 17 parcels and liberal mask "
    f" is located at: {yeo['thick_17']}"
)

# %%
# Load functional data
# --------------------
data = datasets.fetch_development_fmri(n_subjects=10)

print(
    "Functional nifti images (4D, e.g., one subject) "
    f"are located at : {data.func[0]!r}"
)
print(
    "Counfound csv files (of same subject) are located "
    f"at : {data['confounds'][0]!r}"
)


# %%
# Extract coordinates on Yeo atlas - parcellations
# ------------------------------------------------
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker

# ConenctivityMeasure from Nilearn uses simple 'correlation' to compute
# connectivity matrices for all subjects in a list
connectome_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)

# create masker using MultiNiftiLabelsMasker to extract functional data within
# atlas parcels from multiple subjects using parallelization to speed up the
# computation
masker = MultiNiftiLabelsMasker(
    labels_img=yeo["thick_17"],  # Both hemispheres
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    n_jobs=2,
)

# extract time series from all subjects
time_series = masker.fit_transform(data.func, confounds=data.confounds)

# calculate correlation matrices across subjects and display
correlation_matrices = connectome_measure.fit_transform(time_series)

# Mean correlation matrix across 10 subjects can be grabbed like this,
# using connectome measure object
mean_correlation_matrix = connectome_measure.mean_

# useful for plotting connectivity interactions on glass brain
from nilearn import plotting

# grab center coordinates for atlas labels
coordinates = plotting.find_parcellation_cut_coords(labels_img=yeo["thick_17"])

# plot connectome with 80% edge strength in the connectivity
left_connectome = plotting.plot_connectome(
    mean_correlation_matrix, coordinates, edge_threshold="80%"
)

# %%
# Note that the approach above will extract time series and compute a
# single connectivity matrix for both hemispheres. However, the connectome
# is plotted only for the left hemisphere. If your aim is to compute and plot
# hemisphere-wise connectivity, you can follow the example below.
# First, create a separate atlas image for each hemisphere:

import nibabel as nb
import numpy as np

from nilearn.image import get_data, new_img_like
from nilearn.image.resampling import coord_transform

# load the atlas image first
label_image = nb.load(yeo["thick_17"])

# extract the affine matrix of the image
labels_affine = label_image.affine

# generate image coordinates using affine
x, y, z = coord_transform(0, 0, 0, np.linalg.inv(labels_affine))

# generate an separate image for the left hemisphere
# left/right split is done along x-axis
left_hemi = get_data(label_image).copy()
left_hemi[: int(x)] = 0
label_image_left = new_img_like(label_image, left_hemi, labels_affine)

# same for the right hemisphere
right_hemi = get_data(label_image).copy()
right_hemi[int(x) :] = 0
label_image_right = new_img_like(label_image, right_hemi, labels_affine)

# %%
# Then, create a masker object, compute a connectivity matrix and
# plot the results for each hemisphere:

for hemi, img in zip(["right", "left"], [label_image_right, label_image_left]):
    masker = MultiNiftiLabelsMasker(
        labels_img=img,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
    )

    time_series = masker.fit_transform(data.func, confounds=data.confounds)

    correlation_matrices = connectome_measure.fit_transform(time_series)
    mean_correlation_matrix = connectome_measure.mean_

    coordinates = plotting.find_parcellation_cut_coords(
        labels_img=img, label_hemisphere=hemi
    )

    plotting.plot_connectome(
        mean_correlation_matrix,
        coordinates,
        edge_threshold="80%",
        title=f"Yeo Atlas 17 thick (func) - {hemi}",
    )

plotting.show()

# %%
# Plot a directed connectome - asymmetric connectivity measure
# ------------------------------------------------------------
# In this section, we use the lag-1 correlation as the connectivity
# measure, which leads to an asymmetric connectivity matrix.
# The plot_connectome function accepts both symmetric and asymmetric
# matrices, but plot the latter as a directed graph.


# Define a custom function to compute lag correlation on the time series
def lag_correlation(time_series, lag):
    n_subjects = len(time_series)
    _, n_features = time_series[0].shape
    lag_cor = np.zeros((n_subjects, n_features, n_features))
    for subject, serie in enumerate(time_series):
        for i in range(n_features):
            for j in range(n_features):
                if lag == 0:
                    lag_cor[subject, i, j] = np.corrcoef(
                        serie[:, i], serie[:, j]
                    )[0, 1]
                else:
                    lag_cor[subject, i, j] = np.corrcoef(
                        serie[lag:, i], serie[:-lag, j]
                    )[0, 1]
    return np.mean(lag_cor, axis=0)


# Compute lag-0 and lag-1 correlations and plot associated connectomes
for lag in [0, 1]:
    lag_correlation_matrix = lag_correlation(time_series, lag)
    plotting.plot_connectome(
        lag_correlation_matrix,
        coordinates,
        edge_threshold="90%",
        title=f"Lag-{lag} correlation",
    )

# %%
# Load probabilistic atlases - extracting coordinates on brain maps
# -----------------------------------------------------------------

dim = 64
difumo = datasets.fetch_atlas_difumo(
    dimension=dim, resolution_mm=2, legacy_format=False
)

# %%
# Iterate over fetched atlases to extract coordinates - probabilistic
# -------------------------------------------------------------------
from nilearn.maskers import MultiNiftiMapsMasker

# create masker using MultiNiftiMapsMasker to extract functional data within
# atlas parcels from multiple subjects using parallelization to speed up the
# # computation
masker = MultiNiftiMapsMasker(
    maps_img=difumo.maps,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    n_jobs=2,
)

# extract time series from all subjects
time_series = masker.fit_transform(data.func, confounds=data.confounds)

# calculate correlation matrices across subjects and display
correlation_matrices = connectome_measure.fit_transform(time_series)

# Mean correlation matrix across 10 subjects can be grabbed like this,
# using connectome measure object
mean_correlation_matrix = connectome_measure.mean_

# grab center coordinates for probabilistic atlas
coordinates = plotting.find_probabilistic_atlas_cut_coords(
    maps_img=difumo.maps
)

# plot connectome with 85% edge strength in the connectivity
plotting.plot_connectome(
    mean_correlation_matrix,
    coordinates,
    edge_threshold="85%",
    title=f"DiFuMo with {dim} dimensions (probabilistic)",
)
plotting.show()
