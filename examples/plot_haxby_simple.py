"""
Simple example of decoding: the Haxby data
==============================================

Here is a simple example of decoding, reproducing the Haxby 2001
study on a face vs cat discrimination task in a mask of the ventral
stream.
"""

### Load haxby dataset ########################################################

from nilearn import datasets
haxby_dataset = datasets.fetch_haxby()

### Load Target labels ########################################################

import numpy as np
# Load target information as string and give a numerical identifier to each
labels = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")

# scikit-learn >= 0.14 supports text labels. You can replace this line by:
# target = labels['labels']
_, target = np.unique(labels['labels'], return_inverse=True)

### Keep only data corresponding to faces or cats #############################
condition_mask = np.logical_or(labels['labels'] == 'face',
                               labels['labels'] == 'cat')
target = target[condition_mask]


### Prepare the data: apply the mask ##########################################

from nilearn.input_data import NiftiMasker
mask_filename = haxby_dataset.mask_vt[0]
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True)

func_filename = haxby_dataset.func[0]
# We give the nifti_masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = nifti_masker.fit_transform(func_filename)

# Restrict the classification to the face vs cat discrimination
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
coef_img = nifti_masker.inverse_transform(coef_)

# Save the coefficients as a Nifti image
coef_img.to_filename('haxby_svc_weights.nii')

### Visualization #############################################################
import pylab as plt
from nilearn.image.image import mean_img
from nilearn.plotting import plot_roi, plot_stat_map

mean_epi = mean_img(func_filename)
plot_stat_map(coef_img, mean_epi, title="SVM weights", display_mode="yx")

plot_roi(nifti_masker.mask_img_, mean_epi, title="Mask", display_mode="yx")

plt.show()

