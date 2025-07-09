"""
Extracting signals of a probabilistic atlas of functional regions
=================================================================

This example extracts the signal on regions defined via a probabilistic
atlas, to construct a functional connectome.

We use the `MSDL atlas
<https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/>`_
of functional regions in movie-watching.

The key to extract signals is to use the
:class:`~nilearn.maskers.NiftiMapsMasker` that can transform nifti
objects to time series using a probabilistic atlas.

As the MSDL atlas comes with (x, y, z) :term:`MNI` coordinates for the
different regions, we can visualize the matrix as a graph of
interaction in a brain. To avoid having too dense a graph, we represent
only the 20% edges with the highest values.

.. include:: ../../../examples/masker_note.rst

"""

# %%
# Retrieve the atlas and the data
# -------------------------------
from nilearn.datasets import fetch_atlas_msdl, fetch_development_fmri

atlas = fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# Load the functional datasets
data = fetch_development_fmri(n_subjects=1)

print(
    "First subject resting-state nifti image (4D) is located "
    f"at: {data.func[0]}"
)

# %%
# Extract the time series
# -----------------------
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds=True,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
time_series = masker.fit_transform(data.func[0], confounds=data.confounds)

# %%
# We can generate an HTML report and visualize the components of the
# :class:`~nilearn.maskers.NiftiMapsMasker`.
# You can pass the indices of the spatial maps you want to include in the
# report in the order you want them to appear.
# Here, we only include maps 2, 6, 7, 16, and 21 in the report:
report = masker.generate_report(displayed_maps=[2, 6, 7, 16, 21])
report

# %%
# `time_series` is now a 2D matrix, of shape (number of time points x
# number of regions)
print(time_series.shape)

# %%
# Build and display a correlation matrix
# --------------------------------------
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Display the correlation matrix
import numpy as np

from nilearn.plotting import plot_connectome, plot_matrix, show

# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)
plot_matrix(correlation_matrix, labels=labels, vmax=0.8, vmin=-0.8)
# %%
# And now display the corresponding graph
# ---------------------------------------
coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plot_connectome(correlation_matrix, coords, edge_threshold="80%")

show()

# %%
# 3D visualization in a web browser
# ---------------------------------
# An alternative to :func:`~nilearn.plotting.plot_connectome` is to use
# :func:`~nilearn.plotting.view_connectome` that gives more interactive
# visualizations in a web browser. See :ref:`interactive-connectome-plotting`
# for more details.
from nilearn.plotting import view_connectome

view = view_connectome(correlation_matrix, coords, edge_threshold="80%")

# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view

# %%

# uncomment this to open the plot in a web browser:
# view.open_in_browser()

# sphinx_gallery_dummy_images=2
