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
# XXX: must get the code to run for more than 1 subject

### Preprocess ################################################################
from nisl import io

masker = io.NiftiMultiMasker(smooth=8, detrend=True)
data_masked = masker.fit_transform(dataset.func)

# Concatenate all the subjects
#fmri_data = np.concatenate(data_masked, axis=1)
fmri_data = data_masked

mean_epi = masker.inverse_transform(fmri_data[0].mean(axis=0)).get_data()

### Apply ICA #################################################################

from nisl.decomposition.canica import CanICA
n_components = 20
ica = CanICA(n_components=n_components, random_state=42)
components_masked = ica.fit(data_masked).maps_

# We normalize the estimated components, for thresholding to make sens
# components_masked -= components_masked.mean(axis=0)
# components_masked /= components_masked.std(axis=0)
# Threshold
# components_masked[np.abs(components_masked) < .5] = 0

# Now we inverting the masking operation, to go back to a full 3D
# representation
components_img = masker.inverse_transform(components_masked)
components = components_img.get_data()

# Using a masked array is important to have transparency in the figures
components = np.ma.masked_equal(components, 0, copy=False)

### Visualize the results #####################################################
# Show some interesting components
# mean_epi = mean_img.get_data()
import pylab as pl
pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 20, 16]))
pl.imshow(np.rot90(mean_epi[:, :, 20]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(components[:, :, 20, 16]), interpolation='nearest',
          cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)

pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 25, 19]))
pl.imshow(np.rot90(mean_epi[:, :, 25]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(components[:, :, 25, 19]), interpolation='nearest',
           cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)
pl.show()
