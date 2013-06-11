"""
Group analysis of resting-state fMRI with ICA: CanICA
=====================================================

An example applying CanICA to resting-state data. This example applies it
to 40 subjects of the ADHD200 datasets.

CanICA is an ICA method for group-level analysis of fMRI data. Compared
to other strategies, it brings a well-controlled group model, as well as a
thresholding algorithm controlling for specificity and sensitivity with
an explicit model of the signal. The reference papers are:

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177

Pre-prints for both papers are available on hal
(http://hal.archives-ouvertes.fr)
"""
import numpy as np

### Load ADHD rest dataset ####################################################
from nisl import datasets
# Here we use a limited number of subjects to get faster-running code. For
# better results, simply increase the number.
dataset = datasets.fetch_adhd()
func_files = dataset.func[:5]

### Preprocess ################################################################
from nisl import io

# This is a multi-subject method, thus we need to use the
# NiftiMultiMasker, rather than the NiftiMasker
# We specify the target_affine to downsample to 3mm isotropic
# resolution
masker = io.NiftiMultiMasker(smoothing_fwhm=6,
                             target_affine=np.diag((3, 3, 3)),
                             memory="nisl_cache", memory_level=1,
                             verbose=True)
data_masked = masker.fit_transform(func_files)

mean_epi = masker.inverse_transform(data_masked[0].mean(axis=0)).get_data()

### Apply CanICA ##############################################################

from nisl.decomposition.old_canica import CanICA
n_components = 20
ica = CanICA(n_components=n_components, random_state=42, memory="nisl_cache",
             maps_only=True)
components_masked = ica.fit(data_masked).components_

# We normalize the estimated components, for thresholding to make sense
# XXX: this should probably be integrated in the CanICA object
components_masked -= components_masked.mean(axis=1)[:, np.newaxis]
components_masked /= components_masked.std(axis=1)[:, np.newaxis]
# Threshold
#threshold = (stats.norm.isf(0.5*threshold_p_value)
#                                 /np.sqrt(components_masked.shape[0]))
threshold = .9
components_masked[np.abs(components_masked) < threshold] = 0

# Now invert the masking operation, to go back to a full 3D
# representation
components_img = masker.inverse_transform(components_masked)
components = components_img.get_data()

# Using a masked array is important to have transparency in the figures
components = np.ma.masked_equal(components, 0, copy=False)

### Visualize the results #####################################################
# Show some interesting components
import pylab as pl
from scipy import ndimage

for i in range(n_components):
    pl.figure()
    pl.axis('off')
    cut_coord = ndimage.maximum_position(np.abs(components[..., i]))[2]
    vmax = np.max(np.abs(components[:, :, cut_coord, i]))
    pl.imshow(np.rot90(mean_epi[:, :, cut_coord]), interpolation='nearest',
              cmap=pl.cm.gray)
    pl.imshow(np.rot90(components[:, :, cut_coord, i]),
              interpolation='nearest', cmap=pl.cm.jet, vmax=vmax, vmin=-vmax)

pl.show()
