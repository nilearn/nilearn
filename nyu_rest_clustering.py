### Options ###################################################################

algorithm = ['ward', 'spectral_clustering']
generate_image = [None, 'picture.png']
n_clusters = [None] + range(4, 20, 2)

### Init ######################################################################

algorithm = 'ward'
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
m = mask.compute_mask(mean_img)
X_masked = X[m]

### Apply clustering ##########################################################

if n_clusters is None:
    n_clusters = 2

if algorithm == 'ward':
    s = m.shape
    connectivity = image.img_to_graph(X[:, :, :, 0], mask=m)
    ward = Ward(n_clusters=n_clusters, connectivity=connectivity.tolil())
    ward.fit(X_masked)
    L = - np.ones(s)
    L[m] = ward.labels_
    plt.imshow(L[:, :, 20], interpolation='nearest', cmap=plt.cm.spectral)
elif algorithm == 'spectral_clustering':
    from sklearn.cluster import spectral_clustering
    X = dataset.func[0][:, :, :, 0]
    graph = image.img_to_graph(X, mask=m)
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
