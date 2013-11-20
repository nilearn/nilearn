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
import nibabel

### Load ADHD rest dataset ####################################################
from nilearn import datasets
# Here we use a limited number of subjects to get faster-running code. For
# better results, simply increase the number.

dataset = datasets.fetch_adhd()
func_files = dataset.func

### Filter and mask ###########################################################
from nilearn.image import resample_img

# This is a multi-subject method, thus we need to use the
# MultiNiftiMasker, rather than the NiftiMasker

epi_img = nibabel.load(func_files[0])
mean_epi = epi_img.get_data().mean(axis=-1)
mean_epi_img = nibabel.Nifti1Image(mean_epi, epi_img.get_affine())
mean_epi = resample_img(mean_epi_img).get_data()

### Apply CanICA ##############################################################
from nilearn.decomposition.canica import CanICA

n_components = 20
canica = CanICA(n_components=n_components,
                smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                threshold=3., verbose=10, random_state=0)
canica.fit(func_files)

components = canica.masker_.inverse_transform(canica.components_).get_data()

### Visualize the results #####################################################
# Show some interesting components
import pylab as pl
from scipy import ndimage

# Using a masked array is important to have transparency in the figures
components = np.ma.masked_equal(components, 0, copy=False)

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
