### Load nyu_rest dataset #####################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest()

### Mask ######################################################################

from nisl import masking
import numpy as np
X = dataset.func[0]
# Calculate the mean of all images to compute the mask
mean_img = np.mean(X, axis=3)
mask = masking.compute_mask(mean_img)
# Mask data
X_masked = X[mask]

### Ward ######################################################################

# Compute connectivty map
from sklearn.feature_extraction import image
s = mask.shape
c = image.grid_to_graph(n_x=s[0], n_y=s[1], n_z=s[2], mask=mask)

# Computing the ward for the first time, this is long...
from sklearn.cluster import WardAgglomeration
import time
start = time.time()
ward = WardAgglomeration(n_clusters=500, connectivity=c, memory='ward')
ward.fit(X_masked.T)
print "Ward agglomeration 500 clusters: %.2fs" % (time.time() - start)

# Compute the ward with more clusters, should be faster
start = time.time()
ward = WardAgglomeration(n_clusters=1000, connectivity=c, memory='ward')
ward.fit(X_masked.T)
print "Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start)

### Prepare output ############################################################

# Unmask data
L = - np.ones(X[:, :, :, 0].shape)
L[mask] = ward.labels_

# Create a compressed picture
X_r = ward.transform(X_masked.T)
X_c = ward.inverse_transform(X_r)
C = - np.ones(X[:, :, :, 0].shape)
C[mask] = X_c[0]

### Show result ###############################################################

from matplotlib import pyplot as plt
plt.figure()
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(X[..., 20, 0], interpolation='nearest', cmap=plt.cm.spectral)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(L[:, :, 20], interpolation='nearest', cmap=plt.cm.spectral)
plt.title('Labels')
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(C[:, :, 20], interpolation='nearest', cmap=plt.cm.spectral)
plt.title('Compressed representation')
plt.show()
