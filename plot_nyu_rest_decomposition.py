import numpy as np

### Load nyu_trt dataset ######################################################
from nisl import datasets
dataset = datasets.fetch_nyu_rest(n_subjects=3)

### Preprocess ################################################################

fmri_data = np.concatenate((
                            dataset.func[0],
                            dataset.func[1],
                            dataset.func[2]),
                            axis=3)

# Apply a small amount of Gaussian smoothing
from scipy import ndimage
for image in fmri_data.T:
    image[...] = ndimage.gaussian_filter(image, 1.5)

mean_img = np.mean(fmri_data, axis=3)

# Mask non brain areas
from nisl import masking
mask = masking.compute_mask(mean_img)
data_masked = fmri_data[mask]


### Apply ICA #################################################################

from sklearn.decomposition import FastICA
n_components = 20
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit(data_masked).transform(data_masked)
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)

(x, y, z) = mean_img.shape
components = np.zeros((x, y, z, n_components))
components[mask] = components_masked

# Threshold
components[np.abs(components) < .5] = 0
components = np.ma.masked_equal(components, 0, copy=False)

# Show some interesting components
import pylab as pl
pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 20, 16]))
pl.imshow(mean_img[:, :, 20], interpolation='nearest', cmap=pl.cm.gray)
pl.imshow(components[:, :, 20, 16], interpolation='nearest',
           cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)
pl.title('Default mode')

pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 25, 19]))
pl.imshow(mean_img[:, :, 25], interpolation='nearest', cmap=pl.cm.gray)
pl.imshow(components[:, :, 25, 19], interpolation='nearest',
           cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)
pl.title('Ventral attention network')
pl.show()
