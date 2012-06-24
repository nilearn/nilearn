"""
Learning a brain parcellation from rest fMRI
=============================================
"""

### Load nyu_rest dataset #####################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest(n_subjects=1)

### Mask ######################################################################

epi_img = dataset.func[0]
# Compute the mask

from nisl import masking
mask = masking.compute_mask(epi_img)
# Mask data
epi_masked = epi_img[mask]

### Ward ######################################################################

# Compute connectivity matrix
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
ward.fit(epi_masked.T)
print "Ward agglomeration 500 clusters: %.2fs" % (time.time() - start)

# Compute the ward with more clusters, should be faster
start = time.time()
ward = WardAgglomeration(n_clusters=1000, connectivity=connectivity,
                         memory='nisl_cache')
ward.fit(epi_masked.T)
print "Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start)

### Prepare output ############################################################

### Show result ###############################################################

from matplotlib import pyplot as pl
pl.figure()

# Display the labels
# Unmask data
import numpy as np
labels = - np.ones(mask.shape)
labels[mask] = ward.labels_

cut = labels[:, :, 20].astype(np.int)
colors = np.random.random(size=(ward.n_clusters + 1, 3))
colors[-1] = 0
pl.axis('off')
pl.imshow(colors[cut], interpolation='nearest')
pl.title('Ward parcellation')

# Display the original data
pl.figure()
first_epi_img = epi_img[..., 0].copy()
first_epi_img[np.logical_not(mask)] = 0
pl.imshow(first_epi_img[..., 20], interpolation='nearest',
           cmap=pl.cm.spectral)
pl.axis('off')
pl.title('Original')

# Display the corresponding data compressed using the parcellation
X_r = ward.transform(epi_masked.T)
X_c = ward.inverse_transform(X_r)
compressed_img = np.zeros(mask.shape)
compressed_img[mask] = X_c[0]

pl.figure()
pl.imshow(compressed_img[:, :, 20], interpolation='nearest',
           cmap=pl.cm.spectral)
pl.title('Compressed representation')
pl.axis('off')
pl.show()
