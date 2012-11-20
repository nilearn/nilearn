"""
Simple example of Nifti Masker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

### Load nyu_rest dataset #####################################################

from nisl import datasets
from nisl.io import NiftiMasker
dataset = datasets.fetch_nyu_rest(n_subjects=1)

### Compute the mask ##########################################################

nifti_masker = NiftiMasker(transpose=True)
nifti_masker.fit(dataset.func[0])
mask = nifti_masker.mask_img_.get_data()

### Visualize the mask ########################################################
import pylab as pl
import numpy as np
import nibabel
pl.figure()
pl.axis('off')
pl.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 20, 0]),
          interpolation='nearest', cmap=pl.cm.gray)
ma = np.ma.masked_equal(mask, False)
pl.imshow(np.rot90(ma[..., 20]), interpolation='nearest', cmap=pl.cm.autumn,
          alpha=0.5)
pl.title("Mask")
pl.show()

### Preprocess data ###########################################################
nifti_masker.fit(dataset.func[0])
fmri_masked = nifti_masker.transform(dataset.func[0])

### Run an algorithm ##########################################################
from sklearn.decomposition import FastICA
n_components = 20
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(fmri_masked)

### Reverse masking ###########################################################
components = nifti_masker.inverse_transform(components_masked)

### Show results ##############################################################
components = np.ma.masked_equal(components.get_data(), 0)
pl.figure()
pl.axis('off')
pl.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 20, 0]),
          interpolation='nearest', cmap=pl.cm.gray)
pl.imshow(np.rot90(components[..., 20, 16]), interpolation='nearest',
          cmap=pl.cm.hot)
pl.show()

### The same with a pipeline ##################################################
from sklearn.pipeline import Pipeline
mask_ica = Pipeline([('masking', nifti_masker), ('ica', ica)])
components = nifti_masker.inverse_transform(
    mask_ica.fit_transform(dataset.func[0]))
