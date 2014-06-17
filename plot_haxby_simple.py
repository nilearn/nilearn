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

### Keep only data corresponding to faces or cat ##############################
condition_mask = np.logical_or(labels['labels'] == 'face',
                               labels['labels'] == 'cat')
target = target[condition_mask]


### Prepare the data: apply the mask ##########################################

from nilearn.input_data import NiftiMasker
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask=data.mask_vt[0], standardize=True)

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
import pylab as plt
from nilearn.image.image import mean_img
from nilearn.plotting import plot_roi, plot_stat_map

mean_epi = mean_img(data.func[0])
plot_stat_map(coef_niimg, mean_epi, title="SVM weights")

plot_roi(nifti_masker.mask_img_, mean_epi, title="Mask")

plt.show()

