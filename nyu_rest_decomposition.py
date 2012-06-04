### Options ###################################################################

### Init ######################################################################

n_components = 42
threshold = 5e-3

### Imports ###################################################################

from matplotlib import pyplot as plt
import numpy as np

### Load nyu_trt dataset ######################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest()

### Preprocess ################################################################

# Mask non brain areas
from nisl import mask
X = np.concatenate((
    dataset.func[0],
    dataset.func[1],
    dataset.func[2]),
    axis=3)

mean_img = np.mean(X, axis=3)
mask = mask.compute_mask(mean_img)
X_masked = X[mask]

### Apply requested algorithm #################################################

plt.axis('off')

from sklearn.decomposition import FastICA
X_masked_shape = X_masked.shape
ica = FastICA(n_components=n_components)
S_masked = ica.fit(X_masked).transform(X_masked)
(x, y, z) = mean_img.shape
S = np.zeros((x, y, z, n_components))
S[mask] = S_masked

# Threshold
S[np.abs(S) < threshold] = 0

S = np.ma.masked_equal(S, 0, copy=False)

plt.imshow(mean_img[:, :, 20], interpolation='nearest', cmap=plt.cm.gray)
plt.imshow(S[:, :, 20, 0], interpolation='nearest', cmap=plt.cm.hot)

plt.show()
