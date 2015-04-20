"""
Extracting signals of a probabilistic atlas of rest functional regions
========================================================================

This example extracts the signal on regions defined via a probabilistic
atlas, to construct a functional connectome.

We use the `MSDL atlas
<https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/>`_
of functional regions in rest.

The key to extract signals is to use the
:class:`nilearn.input_data.NiftiMapsMasker` that can transform nifti
objects to time series using a probabilistic atlas.

As the MSDL atlas comes with (x, y, z) MNI coordinates for the different
regions, we can visualize the matrix as a graph of interaction in a
brain. To avoid having too dense a graph, we represent only the 20% edges
with the highest values.

"""

from nilearn import datasets
atlas = datasets.fetch_msdl_atlas()
atlas_filename = atlas['maps']

# Load the labels
import numpy as np
csv_filename = atlas['labels']

# The recfromcsv function can load a csv file
labels = np.recfromcsv(csv_filename)
names = labels['name']

from nilearn.input_data import NiftiMapsMasker
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)

data = datasets.fetch_adhd(n_subjects=1)

# print basic dataset information
print('First subject resting-state nifti image (4D) is located at: %s' %
      data.func[0])

time_series = masker.fit_transform(data.func[0],
                                   confounds=data.confounds)

correlation_matrix = np.corrcoef(time_series.T)

# Display the correlation matrix
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(correlation_matrix, interpolation="nearest")
# And display the labels
x_ticks = plt.xticks(range(len(names)), names, rotation=90)
y_ticks = plt.yticks(range(len(names)), names)

# And now display the corresponding graph
from nilearn import plotting
coords = np.vstack((labels['x'], labels['y'], labels['z'])).T

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="80%")

plt.show()


