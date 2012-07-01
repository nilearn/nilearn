"""
Hierachical clustering to learn a brain parcellation from rest fMRI
====================================================================

We use spatial-constrained Ward-clustering to create a set of
parcels. These parcels are particularly interesting for creating a
'compressed' representation of the data, replacing the data in the fMRI
images by mean on the parcellation.

This parcellation may be useful in a supervised learning, see for
instance: `A supervised clustering approach for fMRI-based inference of
brain states <http://hal.inria.fr/inria-00589201>`_, Michel et al,
Pattern Recognition 2011.

"""

### Load nyu_rest dataset #####################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest(n_subjects=1)

### Mask ######################################################################

fmri_data = dataset.func[0]

# Compute a brain mask
from nisl import masking
mask = masking.compute_mask(fmri_data)

# Mask data: go from a 4D dataset to a 2D dataset with only the voxels
# in the mask
fmri_masked = fmri_data[mask]

### Ward ######################################################################

# Compute connectivity matrix: which voxel is connected to which
from sklearn.feature_extraction import image
shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)

# Computing the ward for the first time, this is long...
from sklearn.cluster import WardAgglomeration
import time
start = time.time()
ward = WardAgglomeration(n_clusters=500, connectivity=connectivity,
                         memory='nisl_cache')
ward.fit(fmri_masked.T)
print "Ward agglomeration 500 clusters: %.2fs" % (time.time() - start)

# Compute the ward with more clusters, should be faster as we are using
# the caching mechanism
start = time.time()
ward = WardAgglomeration(n_clusters=1000, connectivity=connectivity,
                         memory='nisl_cache')
ward.fit(fmri_masked.T)
print "Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start)

### Show result ###############################################################

# Unmask data
import numpy as np
labels = - np.ones(mask.shape)
labels[mask] = ward.labels_

# Display the labels
import pylab as pl

# Cut at z=20
cut = labels[:, :, 20].astype(np.int)
# Assign random colors to each cluster. For this we build a random
# RGB look up table associating a color to each cluster, and apply it
# below
colors = np.random.random(size=(ward.n_clusters + 1, 3))
# Cluster '-1' should be black (it's outside the brain)
colors[-1] = 0
pl.figure()
pl.axis('off')
pl.imshow(colors[np.rot90(cut)], interpolation='nearest')
pl.title('Ward parcellation')

# Display the original data
pl.figure()
first_fmri_img = fmri_data[..., 0].copy()
# Outside the mask: a uniform value, smaller than inside the mask
first_fmri_img[np.logical_not(mask)] = 0.9*first_fmri_img[mask].min()
vmax = first_fmri_img[..., 20].max()
pl.imshow(np.rot90(first_fmri_img[..., 20]),
          interpolation='nearest', cmap=pl.cm.spectral, vmax=vmax)
pl.axis('off')
pl.title('Original')

# A reduced data can be create by taking the parcel-level average:
# Note that, as many objects in the scikit-learn, the ward object exposes
# a transform method that modifies input features. Here it reduces their
# dimension
fmri_reduced = ward.transform(fmri_masked.T)

# Display the corresponding data compressed using the parcellation
fmri_compressed = ward.inverse_transform(fmri_reduced)
compressed_img = first_fmri_img.copy()
compressed_img[mask] = fmri_compressed[0]

pl.figure()
pl.imshow(np.rot90(compressed_img[:, :, 20]),
           interpolation='nearest', cmap=pl.cm.spectral, vmax=vmax)
pl.title('Compressed representation')
pl.axis('off')
pl.show()
