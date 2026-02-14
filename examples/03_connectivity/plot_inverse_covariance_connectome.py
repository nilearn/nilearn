"""
Computing a connectome with sparse inverse covariance
=====================================================

This example constructs a functional connectome using the sparse inverse
covariance.

We use the `MSDL atlas
<https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/>`_
of functional regions in movie watching, and the
:class:`~nilearn.maskers.NiftiMapsMasker` to extract time series.

Note that the inverse covariance (or precision) contains values that can
be linked to *negated* partial correlations, so we negated it for
display.

As the MSDL atlas comes with (x, y, z) :term:`MNI` coordinates for
the different regions, we can visualize the matrix as a graph of
interaction in a brain. To avoid having too dense a graph, we
represent only the 20% edges with the highest values.

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

# Loading the functional datasets
data = fetch_development_fmri(n_subjects=1)

# print basic information on the dataset
print(f"First subject functional nifti images (4D) are at: {data.func[0]}")

# %%
# Extract time series
# -------------------
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize_confounds=True,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

time_series = masker.fit_transform(data.func[0], confounds=data.confounds)

# %%
# Compute the sparse inverse covariance
# -------------------------------------
from sklearn.covariance import GraphicalLassoCV

estimator = GraphicalLassoCV(verbose=True)
estimator.fit(time_series)

# %%
# Display the connectome matrix
# -----------------------------
from nilearn.plotting import (
    plot_connectome,
    plot_matrix,
    show,
    view_connectome,
)

# Display the covariance

# The covariance can be found at estimator.covariance_
plot_matrix(
    estimator.covariance_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Covariance",
)

# %%
# And now display the corresponding graph
# ---------------------------------------
coords = atlas.region_coords

plot_connectome(estimator.covariance_, coords, title="Covariance")


# %%
# Display the sparse inverse covariance
# -------------------------------------
# we negate it to get partial correlations
plot_matrix(
    -estimator.precision_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Sparse inverse covariance",
)

# %%
# And now display the corresponding graph
# ----------------------------------------
plot_connectome(
    -estimator.precision_, coords, title="Sparse inverse covariance"
)

show()

# %%
# 3D visualization in a web browser
# ---------------------------------
# An alternative to :func:`~nilearn.plotting.plot_connectome` is to use
# :func:`~nilearn.plotting.view_connectome` that gives more interactive
# visualizations in a web browser. See :ref:`interactive-connectome-plotting`
# for more details.


view = view_connectome(-estimator.precision_, coords)

# In a notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view

# %%

# uncomment this to open the plot in a web browser:
# view.open_in_browser()

# sphinx_gallery_dummy_images=4
