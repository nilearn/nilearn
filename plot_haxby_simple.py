"""
Simple example of decoding: the Haxby dataset
==============================================

Here is a simple example of decoding, reproducing the Haxby 2001
study.
"""

### Load haxby dataset ########################################################

from nisl import datasets
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

### Load the mask #############################################################

from nisl.io import NiftiMasker
nifti_masker = NiftiMasker(mask=dataset.mask_vt[0])

# We give to the nifti_masker a filename, as retrieve a 2D array ready
# for machine learing with scikit-learn
fmri_masked = nifti_masker.fit_transform(dataset.func[0])

### Prediction function #######################################################

# First, we remove rest condition
fmri_masked = fmri_masked[no_rest_indices]

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
import pylab as pl
import nibabel

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


### Visualize the mask ########################################################

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

