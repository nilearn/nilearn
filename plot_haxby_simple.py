"""
Simple example of decoding: the Haxby dataset
==============================================

Here is a simple example of decoding, reproducing the Haxby 2001
study.
"""

### Load haxby dataset ########################################################

from nilearn import datasets
dataset = datasets.fetch_haxby()

### Load Target labels ########################################################

import numpy as np
import sklearn.utils.fixes
# Load target information as string and give a numerical identifier to each
labels = np.loadtxt(dataset.session_target[0], dtype=np.str, skiprows=1,
                    usecols=(0,))

# For compatibility with numpy 1.3 and scikit-learn 0.12
# "return_inverse" option appeared in numpy 1.4, scikit-learn >= 0.14 supports
# text labels.
# With scikit-learn >= 0.14, replace this line by: target = labels
_, target = sklearn.utils.fixes.unique(labels, return_inverse=True)

### Keep only data corresponding to faces or houses ###########################
condition_mask = np.logical_or(labels == 'face', labels == 'house')
target = target[condition_mask]


### Load the mask #############################################################

from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(mask=dataset.mask_vt[0])

# We give the nifti_masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = nifti_masker.fit_transform(dataset.func[0])

### Prediction function #######################################################

# First, we narrow to the face vs house classification
fmri_masked = fmri_masked[condition_mask]

# Here we use a Support Vector Classification, with a linear kernel and C=1
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=1.)

# And we run it
svc.fit(fmri_masked, target)
y_pred = svc.predict(fmri_masked)

### Unmasking #################################################################

# Look at the discriminating weights
coef_ = svc.coef_

# Reverse masking thanks to the Nifti Masker
niimg = nifti_masker.inverse_transform(coef_)

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
pl.imshow(np.rot90(act[..., 27, 0]), cmap=pl.cm.hot,
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

