"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""

import numpy as np

### Load nyu_rest dataset #####################################################
from nilearn import datasets
# Here we use only 3 subjects to get faster-running code. For better
# results, simply increase this number
dataset = datasets.fetch_nyu_rest(n_subjects=1)
# XXX: must get the code to run for more than 1 subject

### Preprocess ################################################################
from nilearn.input_data import NiftiMasker

masker = NiftiMasker(smoothing_fwhm=8, memory='nilearn_cache', memory_level=1,
                        standardize=False)
data_masked = masker.fit_transform(dataset.func[0])

# Concatenate all the subjects
#fmri_data = np.concatenate(data_masked, axis=1)
fmri_data = data_masked

# Take the mean along axis 3: the direction of time
mean_img = masker.inverse_transform(fmri_data.mean(axis=0))


### Apply ICA #################################################################

from sklearn.decomposition import FastICA
n_components = 20
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(data_masked.T).T

# Normalize estimated components, for thresholding to make sense
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)
# Threshold
components_masked[components_masked < .8] = 0

# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)
components = component_img.get_data()

# Using a masked array is important to have transparency in the figures
components = np.ma.masked_equal(components, 0, copy=False)

### Visualize the results #####################################################
# Show some interesting components
mean_epi = mean_img.get_data()
import pylab as pl
pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 25, 5]))
pl.imshow(np.rot90(mean_epi[:, :, 25]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(components[:, :, 25, 5]), interpolation='nearest',
          cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)

pl.figure()
pl.axis('off')
vmax = np.max(np.abs(components[:, :, 23, 12]))
pl.imshow(np.rot90(mean_epi[:, :, 23]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(components[:, :, 23, 12]), interpolation='nearest',
          cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)
pl.show()
