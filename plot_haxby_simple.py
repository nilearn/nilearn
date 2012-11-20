"""
Simple example of Nifti Masker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

### Load haxby dataset ########################################################

from nisl import datasets
from nisl.io import NiftiMasker
dataset = datasets.fetch_haxby()

### Load Target labels ########################################################

import numpy as np

# Load target information as string and give a numerical identifier to each
labels = np.loadtxt(dataset.session_target[0], dtype=np.str, skiprows=1,
                    usecols=(0,))
index, target = np.unique(labels, return_inverse=True)

### Remove resting state condition ############################################

no_rest_indices = (labels != 'rest')
target = target[no_rest_indices]

### Load and visualize the mask ###############################################

import pylab as pl
import numpy as np
import nibabel

nifti_masker = NiftiMasker(mask=dataset.mask_vt[0])
nifti_masker.fit(dataset.func[0])
mask = nifti_masker.mask_img_.get_data().astype(np.bool)

pl.figure()
pl.axis('off')
pl.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 27, 0]),
          interpolation='nearest', cmap=pl.cm.gray)
ma = np.ma.masked_equal(mask, False)
pl.imshow(np.rot90(ma[..., 27]), interpolation='nearest', cmap=pl.cm.autumn,
          alpha=0.5)
pl.title("Mask")
pl.show()

### Preprocess data ###########################################################

fmri_masked = nifti_masker.transform(dataset.func[0])
# We remove rest condition
fmri_masked = fmri_masked[no_rest_indices]

### Prediction function #######################################################

# Here we use a Support Vector Classification, with a linear kernel and C=1
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=1.)

# And we run it
svc.fit(fmri_masked, target)
y_pred = svc.predict(fmri_masked)

### Unmasking #################################################################

# Look at the discriminating weights
sv = svc.support_vectors_

# Reverse masking thanks to the Nifti Masker
niimg = nifti_masker.inverse_transform(sv[0])

### Visualization #############################################################

# We use a masked array so that the voxels at '-1' are displayed transparently
act = np.ma.masked_array(niimg.get_data(), niimg.get_data() == 0)

### Create the figure
pl.figure()
pl.axis('off')
pl.title('SVM vectors')
pl.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 27, 0]),
          interpolation='nearest', cmap=pl.cm.gray)
pl.imshow(np.rot90(act[..., 27]), cmap=pl.cm.hot,
          interpolation='nearest')
pl.show()
