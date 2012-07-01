"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""

import numpy as np

### Load nyu_rest dataset #####################################################
from nisl import datasets
# Here we use only 3 subjects to get faster-running code. For better
# results, simply increase this number
dataset = datasets.fetch_nyu_rest(n_subjects=3)

### Preprocess ################################################################

# Concatenate all the subjects
fmri_data = np.concatenate(dataset.func, axis=3)

# Apply a small amount of Gaussian smoothing: in the case of ICA it is
# important as it introduces a spatial model that ICA lacks and greatly
# reduces the high-frequency signal
from scipy import ndimage
for image in fmri_data.T:
    # This works efficiently because image is a view on fmri_data
    image[...] = ndimage.gaussian_filter(image, 1.5)

# Take the mean along axis 3: the direction of time
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

# We normalize the estimated components, for thresholding to make sens
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)
# Threshold
components_masked[np.abs(components_masked) < .5] = 0

# Now we inverting the masking operation, to go back to a full 3D
# representation
(x, y, z) = mean_img.shape
components = np.zeros((x, y, z, n_components))
components[mask] = components_masked

# Using a masked array is important to have transparency in the figures
components = np.ma.masked_equal(components, 0, copy=False)

### Visualize the results #####################################################
# Show some interesting components
import pylab as pl
pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 20, 16]))
pl.imshow(np.rot90(mean_img[:, :, 20]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(components[:, :, 20, 16]), interpolation='nearest',
          cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)

pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 25, 19]))
pl.imshow(np.rot90(mean_img[:, :, 25]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(components[:, :, 25, 19]), interpolation='nearest',
           cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)
pl.show()
