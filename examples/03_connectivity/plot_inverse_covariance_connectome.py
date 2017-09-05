"""
Computing a connectome with sparse inverse covariance
=======================================================

This example constructs a functional connectome using the sparse inverse
covariance.

We use the `MSDL atlas
<https://team.inria.fr/parietal/18-2/spatial_patterns/spatial-patterns-in-resting-state/>`_
of functional regions in rest, and the
:class:`nilearn.input_data.NiftiMapsMasker` to extract time series.

Note that the inverse covariance (or precision) contains values that can
be linked to *negated* partial correlations, so we negated it for
display.

As the MSDL atlas comes with (x, y, z) MNI coordinates for the different
regions, we can visualize the matrix as a graph of interaction in a
brain. To avoid having too dense a graph, we represent only the 20% edges
with the highest values.

"""

##############################################################################
# Retrieve the atlas and the data
# --------------------------------
from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

# Loading the functional datasets
data = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti images (4D) are at: %s' %
      data.func[0])  # 4D data

##############################################################################
# Extract time series
# --------------------
from nilearn.input_data import NiftiMapsMasker
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=5)

time_series = masker.fit_transform(data.func[0],
                                   confounds=data.confounds)

##############################################################################
# Compute the sparse inverse covariance
# --------------------------------------
from sklearn.covariance import GraphLassoCV
estimator = GraphLassoCV()

estimator.fit(time_series)

##############################################################################
# Display the connectome matrix
# ------------------------------
from nilearn import plotting
# Display the covariance

# The covariance can be found at estimator.covariance_
plotting.plot_matrix(estimator.covariance_, labels=labels,
                     figure=(9, 7), vmax=1, vmin=-1,
                     title='Covariance')

##############################################################################
# And now display the corresponding graph
# ----------------------------------------
coords = atlas.region_coords

plotting.plot_connectome(estimator.covariance_, coords,
                         title='Covariance')


##############################################################################
# Display the sparse inverse covariance
# --------------------------------------
# we negate it to get partial correlations
plotting.plot_matrix(-estimator.precision_, labels=labels,
                     figure=(9, 7), vmax=1, vmin=-1,
                     title='Sparse inverse covariance')

##############################################################################
# And now display the corresponding graph
# ----------------------------------------
plotting.plot_connectome(-estimator.precision_, coords,
                         title='Sparse inverse covariance')

plotting.show()
