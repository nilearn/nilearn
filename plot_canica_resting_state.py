"""
Independent component analysis of resting-state fMRI
=====================================================

An example applying ICA to resting-state data.
"""

import numpy as np
from scipy import stats

### Load nyu_rest dataset #####################################################
from nisl import datasets
# Here we use only 3 subjects to get faster-running code. For better
# results, simply increase this number
dataset = datasets.fetch_adhd()
func = [dataset.func[i] for i in [0, 16, -1]]
cfds = [dataset.regressor[i] for i in [0, 16, -1]]
### Preprocess ################################################################
from nisl import io

masker = io.NiftiMultiMasker(smooth=8, detrend=True, confounds=cfds)
data_masked = masker.fit_transform(func)

# Concatenate all the subjects
#fmri_data = np.concatenate(data_masked, axis=1)
fmri_data = data_masked

mean_epi = masker.inverse_transform(fmri_data[0].mean(axis=0)).get_data()

### Apply ICA #################################################################

from nisl.decomposition.canica import CanICA
n_components = 40
ica = CanICA(n_components=n_components, random_state=42, memory="canica", maps_only=True)
components_masked = ica.fit(data_masked).maps_

# We normalize the estimated components, for thresholding to make sens
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)
# Threshold
#threshold = (stats.norm.isf(0.5*threshold_p_value)
#                                 /np.sqrt(components_masked.shape[0]))
threshold=.8
components_masked[np.abs(components_masked) < threshold] = 0

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
#pl.figure()
#pl.axis('off')
#vmax = np.max(np.abs(components[:, :, 20, 1]))
#pl.imshow(np.rot90(mean_epi[:, :, 20]), interpolation='nearest',
#          cmap=pl.cm.gray)
#pl.imshow(np.rot90(components[:, :, 20, 1]), interpolation='nearest',
#          cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)
#pl.show()

for i in range(40):
    pl.figure()
    pl.axis('off')
    vmax = np.max(np.abs(components[:, :, 50, i]))
    pl.imshow(np.rot90(mean_epi[:, :, 50]), interpolation='nearest',
              cmap=pl.cm.gray)
    pl.imshow(np.rot90(components[:, :, 50, i]), interpolation='nearest',
              cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)
    pl.show()

