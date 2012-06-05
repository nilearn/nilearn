### Options ###################################################################

### Init ######################################################################

n_components = 20
threshold = 5e-1

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

from scipy import ndimage
for image in X.T:
    image[...] = ndimage.gaussian_filter(image, 1.5)

mask = mask.compute_mask(mean_img)
X_masked = X[mask]

### Apply requested algorithm #################################################

from sklearn.decomposition import FastICA
X_masked_shape = X_masked.shape
ica = FastICA(n_components=n_components, random_state=42)
S_masked = ica.fit(X_masked).transform(X_masked)
S_masked -= S_masked.mean(axis=0)
S_masked /= S_masked.std(axis=0)
(x, y, z) = mean_img.shape

S = np.zeros((x, y, z, n_components))
S[mask] = S_masked

# Threshold
S[np.abs(S) < threshold] = 0

S = np.ma.masked_equal(S, 0, copy=False)

# Show some interesting slices
plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('Default mode')
vmax = np.max(np.abs(S[..., 10]))
plt.imshow(mean_img[:, :, 20], interpolation='nearest', cmap=plt.cm.gray)
plt.imshow(S[:, :, 20, 10], interpolation='nearest', cmap=plt.cm.jet,
    vmax=vmax, vmin=-vmax)
plt.subplot(122)
plt.axis('off')
plt.title('Ventral attention network')
vmax = np.max(np.abs(S[..., 19]))
plt.imshow(mean_img[:, :, 25], interpolation='nearest', cmap=plt.cm.gray)
plt.imshow(S[:, :, 25, 19], interpolation='nearest', cmap=plt.cm.jet,
    vmax=vmax, vmin=-vmax)
plt.show()
