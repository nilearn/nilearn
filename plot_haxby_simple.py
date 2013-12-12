"""
Simple example of decoding: the Haxby dataset
==============================================

Here is a simple example of decoding, reproducing the Haxby 2001
study on a face vs house discrimination task in a mask of the ventral
stream.
"""

### Load haxby dataset ########################################################

from nilearn import datasets
dataset = datasets.fetch_haxby()

### Load Target labels ########################################################

import numpy as np
# Load target information as string and give a numerical identifier to each
labels = np.recfromcsv(dataset.session_target[0], delimiter=" ")

# scikit-learn >= 0.14 supports text labels. You can replace this line by:
# target = labels['labels']
_, target = np.unique(labels['labels'], return_inverse=True)

### Keep only data corresponding to faces or houses ###########################
condition_mask = np.logical_or(labels['labels'] == 'face',
                               labels['labels'] == 'house')
target = target[condition_mask]


### Load the mask #############################################################

from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(mask=dataset.mask_vt[0])

# We give the nifti_masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = nifti_masker.fit_transform(dataset.func[0])

### Prediction function #######################################################

# Restrict the classification to the face vs house discrimination
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
import matplotlib.pyplot as plt
import nibabel

# We use a masked array so that the voxels at '-1' are displayed transparently
act = np.ma.masked_array(niimg.get_data(), niimg.get_data() == 0)

### Create the figure
plt.figure()
plt.axis('off')
plt.title('SVM vectors')
plt.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 27, 0]),
          interpolation='nearest', cmap=plt.cm.gray)
plt.imshow(np.rot90(act[..., 27, 0]), cmap=plt.cm.hot,
          interpolation='nearest')


### Visualize the mask ########################################################

mask = nifti_masker.mask_img_.get_data()

plt.figure()
plt.axis('off')
plt.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 27, 0]),
          interpolation='nearest', cmap=plt.cm.gray)
ma = np.ma.masked_equal(mask, 0)
plt.imshow(np.rot90(ma[..., 27]), interpolation='nearest', cmap=plt.cm.autumn,
          alpha=0.5)
plt.title("Mask")
plt.show()

