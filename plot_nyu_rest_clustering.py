### Imports ###################################################################

from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_extraction import image

### Load nyu_rest dataset #####################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest()

### Mask ######################################################################

from nisl import mask
X = dataset.func[0]
mean_img = np.mean(X, axis=3)
m = mask.compute_mask(mean_img)
X_masked = X[m]

### Ward ######################################################################

# Compute connectivty map
s = m.shape
connectivity = image.grid_to_graph(n_x=s[0], n_y=s[1], n_z=s[2], mask=m)
# Launch the ward
from sklearn.cluster import Ward
n_clusters = 1000
ward = Ward(n_clusters=n_clusters, connectivity=connectivity)
ward.fit(X_masked)
labels = ward.labels_

### Spectral clustering #######################################################

"""
from sklearn.cluster import spectral_clustering
X = dataset.func[0][:, :, :, 0]
graph = image.img_to_graph(X, mask=m)
graph.data = np.exp(-graph.data / graph.data.std())
labels = spectral_clustering(graph, k=n_clusters)
labels = labels.reshape(X.shape)
"""

### Unmask ####################################################################

L = - np.ones(X[:, :, :, 0].shape)
L[m] = labels

### Show result ###############################################################

plt.axis('off')
plt.title('NYU Clustering')
plt.imshow(L[:, :, 20], interpolation='nearest', cmap=plt.cm.spectral)
plt.show()
