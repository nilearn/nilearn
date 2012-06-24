### Load nyu_rest dataset #####################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest()

### Mask ######################################################################

from nisl import masking
import numpy as np
epi_img = dataset.func[0]
# Calculate the mean of all images to compute the mask
mean_img = np.mean(epi_img, axis=3)
mask = masking.compute_mask(mean_img)
# Mask data
epi_masked = epi_img[mask]

### Ward ######################################################################

# Compute connectivty map
from sklearn.feature_extraction import image
shape = mask.shape
c = image.grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)

# Computing the ward for the first time, this is long...
from sklearn.cluster import WardAgglomeration
import time
start = time.time()
ward = WardAgglomeration(n_clusters=500, connectivity=c, memory='ward')
ward.fit(epi_masked.T)
print "Ward agglomeration 500 clusters: %.2fs" % (time.time() - start)

# Compute the ward with more clusters, should be faster
start = time.time()
ward = WardAgglomeration(n_clusters=1000, connectivity=c, memory='ward')
ward.fit(epi_masked.T)
print "Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start)

### Prepare output ############################################################

# Unmask data
labels = - np.ones(mask.shape)
labels[mask] = ward.labels_

# Create a compressed picture
X_r = ward.transform(epi_masked.T)
X_c = ward.inverse_transform(X_r)
C = - np.ones(mask.shape)
C[mask] = X_c[0]

### Show result ###############################################################

from matplotlib import pyplot as plt
plt.figure()
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(epi_img[..., 20, 0], interpolation='nearest', cmap=plt.cm.spectral)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(labels[:, :, 20], interpolation='nearest', cmap=plt.cm.spectral)
plt.title('Labels')
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(C[:, :, 20], interpolation='nearest', cmap=plt.cm.spectral)
plt.title('Compressed representation')
plt.show()
