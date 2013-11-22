"""
Simple example of NiftiMasker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

### Load nyu_rest dataset #####################################################

from nilearn import datasets
from nilearn.input_data import NiftiMasker
dataset = datasets.fetch_nyu_rest(n_subjects=1)

### Compute the mask ##########################################################

nifti_masker = NiftiMasker(standardize=False,
                           memory="nilearn_cache", memory_level=2)
nifti_masker.fit(dataset.func[0])
mask = nifti_masker.mask_img_.get_data()

### Visualize the mask ########################################################
import matplotlib.pyplot as plt
import numpy as np
import nibabel
plt.figure()
plt.axis('off')
plt.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 20, 0]),
          interpolation='nearest', cmap=plt.cm.gray)
ma = np.ma.masked_equal(mask, False)
plt.imshow(np.rot90(ma[..., 20]), interpolation='nearest', cmap=plt.cm.autumn,
          alpha=0.5)
plt.title("Mask")

### Preprocess data ###########################################################
nifti_masker.fit(dataset.func[0])
fmri_masked = nifti_masker.transform(dataset.func[0])

### Run an algorithm ##########################################################
from sklearn.decomposition import FastICA
n_components = 20
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(fmri_masked.T).T

### Reverse masking ###########################################################
components = nifti_masker.inverse_transform(components_masked)

### Show results ##############################################################
components_data = np.ma.masked_equal(components.get_data(), 0)
plt.figure()
plt.axis('off')
plt.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 20, 0]),
          interpolation='nearest', cmap=plt.cm.gray)
plt.imshow(np.rot90(components_data[..., 20, 7]), interpolation='nearest',
          cmap=plt.cm.hot)
plt.show()
