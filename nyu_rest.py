### Options ###################################################################

algorithm = ['ica', 'ward', 'connected_ward', 'spectral_clustering']
generate_image = [None, 'picture.png']
n_clusters = [None] + range(4, 20, 2)

### Init ######################################################################

algorithm = 'spectral_clustering'
generate_image = 'picture.png'
n_clusters = 8

### Imports ###################################################################

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import Ward
from sklearn.feature_extraction import image

### Load nyu_trt dataset ######################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest()

### Preprocess ################################################################

# Mask non brain areas
from nisl import mask
X = dataset.func[0]
mean_img = np.mean(X, axis=3)
mask = mask.compute_mask(mean_img)

### Apply requested algorithm #################################################

# For ward, reset to default value if needed
if n_clusters is None:
    n_clusters = 2

plt.axis('off')

if algorithm == 'ica':
    from sklearn.decomposition import FastICA
    X = dataset.func[0][:, :, :, 0]
    X_shape = X.shape
    X = np.reshape(X.ravel(), (-1, 1))
    ica = FastICA()
    S = ica.fit(X).transform(X)
    S = np.reshape(S.squeeze(), X_shape)
    plt.imshow(S[:, :, 20], interpolation='nearest', cmap=plt.cm.hot)
elif algorithm == 'ward':
    ward = Ward(n_clusters=n_clusters)
    X = dataset.func[0][:, :, 20, 0]
    X_shape = X.shape
    X = np.reshape(X, (-1, 1))
    ward.fit(X)
    # X = X.reshape(X, X_shape)
    L = np.reshape(ward.labels_, X_shape)
    plt.imshow(L, interpolation='nearest', cmap=plt.cm.spectral)
elif algorithm == 'connected_ward':
    X = dataset.func[0][:, :, 20, 0]
    X_masked = X[mask[:, :, 20]]
    X_shape = mask[:, :, 20].shape
    connectivity = image.img_to_graph(X, mask=mask[:, :, 20])
    ward = Ward(n_clusters=n_clusters, connectivity=connectivity)
    X_masked = np.reshape(X_masked, (-1, 1))
    ward.fit(X_masked)
    # X = X.reshape(X, X_shape)
    L = - np.ones(X_shape)
    L[mask[:, :, 20]] = ward.labels_
    plt.imshow(L, interpolation='nearest', cmap=plt.cm.spectral)
elif algorithm == 'spectral_clustering':
    from sklearn.cluster import spectral_clustering
    X = dataset.func[0][:, :, :, 0]
    graph = image.img_to_graph(X, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())
    labels = spectral_clustering(graph, k=n_clusters)
    labels = labels.reshape(X.shape)
    plt.imshow(X[:, :, 20], cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(labels[:, :, 20] == l, contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])
    plt.xticks(())
    plt.yticks(())

if generate_image is not None:
    plt.savefig('test.png')
else:
    plt.show()
