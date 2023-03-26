"""
Extracting signals of a probabilistic atlas of functional regions
=================================================================

This example extracts the signal on regions defined via a probabilistic
atlas, to construct a functional connectome.

We use the `MSDL atlas
<https://team.inria.fr/parietal/18-2/spatial_patterns/spatial-patterns-in-resting-state/>`_
of functional regions in movie-watching.

The key to extract signals is to use the
:class:`nilearn.maskers.NiftiMapsMasker` that can transform nifti
objects to time series using a probabilistic atlas.

As the MSDL atlas comes with (x, y, z) :term:`MNI` coordinates for the
different regions, we can visualize the matrix as a graph of
interaction in a brain. To avoid having too dense a graph, we represent
only the 20% edges with the highest values.

.. include:: ../../../examples/masker_note.rst

"""
############################################################################
# Retrieve the atlas and the data
# -------------------------------
from nilearn import datasets

atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# Load the functional datasets
data = datasets.fetch_development_fmri(n_subjects=1)

print(
    "First subject resting-state nifti image (4D) is located "
    f"at: {data.func[0]}"
)

############################################################################
# Extract the time series
# -----------------------
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize=True,
    memory="nilearn_cache",
    verbose=5,
)
masker.fit(data.func[0])
time_series = masker.transform(data.func[0], confounds=data.confounds)

############################################################################
# We can generate an HTML report and visualize the components of the
# :class:`~nilearn.maskers.NiftiMapsMasker`.
# You can pass the indices of the spatial maps you want to include in the
# report in the order you want them to appear.
# Here, we only include maps 2, 6, 7, 16, and 21 in the report:
report = masker.generate_report(displayed_maps=[2, 6, 7, 16, 21])
report

############################################################################
# `time_series` is now a 2D matrix, of shape (number of time points x
# number of regions)
print(time_series.shape)

############################################################################
# Build and display a correlation matrix
# --------------------------------------
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind="correlation")
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Display the correlation matrix
import numpy as np
from nilearn import plotting

# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)
plotting.plot_matrix(
    correlation_matrix, labels=labels, colorbar=True, vmax=0.8, vmin=-0.8
)
############################################################################
# And now display the corresponding graph
# ---------------------------------------
from nilearn import plotting

coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(
    correlation_matrix, coords, edge_threshold="80%", colorbar=True
)

plotting.show()

##############################################################################
# 3D visualization in a web browser
# ---------------------------------
# An alternative to :func:`nilearn.plotting.plot_connectome` is to use
# :func:`nilearn.plotting.view_connectome` that gives more interactive
# visualizations in a web browser. See :ref:`interactive-connectome-plotting`
# for more details.


view = plotting.view_connectome(
    correlation_matrix, coords, edge_threshold="80%"
)

# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view

##############################################################################

# uncomment this to open the plot in a web browser:
# view.open_in_browser()
