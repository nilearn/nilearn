"""
Simple example of decoding: the Haxby data
==============================================

Here is a simple example of decoding, reproducing the Haxby 2001
study on a face vs house discrimination task in a mask of the ventral
stream.
"""

### Load haxby dataset ########################################################

from nilearn import datasets
data = datasets.fetch_haxby()

### Load Target labels ########################################################

import numpy as np
# Load target information as string and give a numerical identifier to each
labels = np.recfromcsv(data.session_target[0], delimiter=" ")

# scikit-learn >= 0.14 supports text labels. You can replace this line by:
# target = labels['labels']
_, target = np.unique(labels['labels'], return_inverse=True)

### Keep only data corresponding to faces or houses ###########################
condition_mask = np.logical_or(labels['labels'] == 'face',
                               labels['labels'] == 'house')
target = target[condition_mask]


### Prepare the data: apply the mask ##########################################

from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(mask=data.mask_vt[0])

# We give the nifti_masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = nifti_masker.fit_transform(data.func[0])

# Restrict the classification to the face vs house discrimination
fmri_masked = fmri_masked[condition_mask]

### Prediction ################################################################

# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVC
svc = SVC(kernel='linear')

# And we run it
svc.fit(fmri_masked, target)
prediction = svc.predict(fmri_masked)

### Cross-validation ##########################################################

from sklearn.cross_validation import KFold

cv = KFold(n=len(fmri_masked), n_folds=5)
cv_scores = []

for train, test in cv:
    svc.fit(fmri_masked[train], target[train])
    prediction = svc.predict(fmri_masked[test])
    cv_scores.append(np.sum(prediction == target[test])
                     / float(np.size(target[test])))

print cv_scores

### Unmasking #################################################################

# Retrieve the SVC discriminating weights
coef_ = svc.coef_

# Reverse masking thanks to the Nifti Masker
coef_niimg = nifti_masker.inverse_transform(coef_)

# Use nibabel to save the coefficients as a Nifti image
import nibabel
nibabel.save(coef_niimg, 'haxby_svc_weights.nii')

### Visualization #############################################################
import matplotlib.pyplot as plt

### Create the figure and plot the first EPI image as a background
plt.figure(figsize=(3, 5))

epi_img = nibabel.load(data.func[0])
plt.imshow(np.rot90(epi_img.get_data()[..., 27, 0]),
          interpolation='nearest', cmap=plt.cm.gray)

### Plot the SVM weights
weights = coef_niimg.get_data()
# We use a masked array so that the voxels at '-1' are displayed transparently
weights = np.ma.masked_array(weights, weights == 0)

plt.imshow(np.rot90(weights[..., 27, 0]), cmap=plt.cm.hot,
          interpolation='nearest')

plt.axis('off')
plt.title('SVM weights')


### Visualize the mask ########################################################

mask = nifti_masker.mask_img_.get_data()

plt.figure()
plt.axis('off')
plt.imshow(np.rot90(nibabel.load(data.func[0]).get_data()[..., 27, 0]),
          interpolation='nearest', cmap=plt.cm.gray)
ma = np.ma.masked_equal(mask, 0)
plt.imshow(np.rot90(ma[..., 27]), interpolation='nearest', cmap=plt.cm.autumn,
          alpha=0.5)
plt.title("Mask")
plt.show()

