### Options ###################################################################

### Init ######################################################################

### Imports ###################################################################

from matplotlib import pyplot as plt
import numpy as np

### Load nyu_trt dataset ######################################################

from nisl import datasets
dataset = datasets.fetch_nyu_rest()

### Preprocess ################################################################

# Mask non brain areas
from nisl import mask
X = dataset.func[0]
mean_img = np.mean(X, axis=3)
mask = mask.compute_mask(mean_img)
X_masked = X[mask]

### Apply requested algorithm #################################################

plt.axis('off')

from sklearn.decomposition import FastICA
# X = dataset.func[0][:, :, :, 0]
X_masked_shape = X_masked.shape
X_masked = np.reshape(X_masked.ravel(), (-1, 1))
ica = FastICA()
S_masked = ica.fit(X_masked).transform(X_masked)
S_masked = np.reshape(S_masked.squeeze(), X_masked_shape)
S = np.zeros(X.shape)
S[mask] = S_masked
plt.imshow(S[:, :, 20, 0], interpolation='nearest', cmap=plt.cm.hot)

plt.show()
