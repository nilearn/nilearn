"""
Ward clustering to learn a brain parcellation from rest fMRI
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

import numpy as np
from nilearn import datasets, input_data
dataset = datasets.fetch_nyu_rest(n_subjects=1)
nifti_masker = input_data.NiftiMasker(memory='nilearn_cache', memory_level=1,
                              standardize=False)
fmri_masked = nifti_masker.fit_transform(dataset.func[0])
mask = nifti_masker.mask_img_.get_data().astype(np.bool)

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
ward = WardAgglomeration(n_clusters=1000, connectivity=connectivity,
                         memory='nilearn_cache')
ward.fit(fmri_masked)
print "Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start)

# Compute the ward with more clusters, should be faster as we are using
# the caching mechanism
start = time.time()
ward = WardAgglomeration(n_clusters=2000, connectivity=connectivity,
                         memory='nilearn_cache')
ward.fit(fmri_masked)
print "Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start)

### Show result ###############################################################

# Unmask data
# Avoid 0 label
labels = ward.labels_ + 1
labels = nifti_masker.inverse_transform(labels).get_data()
# 0 is the background, putting it to -1
labels = labels - 1

# Display the labels
import pylab as pl

# Cut at z=20
cut = labels[:, :, 20].astype(int)
# Assign random colors to each cluster. For this we build a random
# RGB look up table associating a color to each cluster, and apply it
# below
import numpy as np
colors = np.random.random(size=(ward.n_clusters + 1, 3))
# Cluster '-1' should be black (it's outside the brain)
colors[-1] = 0
pl.figure()
pl.axis('off')
pl.imshow(colors[np.rot90(cut)], interpolation='nearest')
pl.title('Ward parcellation')

# Display the original data
pl.figure()
first_epi = nifti_masker.inverse_transform(fmri_masked[0]).get_data()
first_epi = np.ma.masked_array(first_epi, first_epi == 0)
# Outside the mask: a uniform value, smaller than inside the mask
first_epi[np.logical_not(mask)] = 0.9 * first_epi[mask].min()
vmax = first_epi[..., 20].max()
vmin = first_epi[..., 20].min()
pl.imshow(np.rot90(first_epi[..., 20]),
          interpolation='nearest', cmap=pl.cm.spectral, vmin=vmin, vmax=vmax)
pl.axis('off')
pl.title('Original (%i voxels)' % fmri_masked.shape[1])

# A reduced data can be create by taking the parcel-level average:
# Note that, as many objects in the scikit-learn, the ward object exposes
# a transform method that modifies input features. Here it reduces their
# dimension
fmri_reduced = ward.transform(fmri_masked)

# Display the corresponding data compressed using the parcellation
fmri_compressed = ward.inverse_transform(fmri_reduced)
compressed = nifti_masker.inverse_transform(
    fmri_compressed[0]).get_data()
compressed = np.ma.masked_equal(compressed, 0)


pl.figure()
pl.imshow(np.rot90(compressed[:, :, 20]),
          interpolation='nearest', cmap=pl.cm.spectral, vmin=vmin, vmax=vmax)
pl.title('Compressed representation (2000 parcels)')
pl.axis('off')
pl.show()
