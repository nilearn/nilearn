"""
Simple example of Nifti Masker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

### Load nyu_rest dataset #####################################################

from nisl import datasets
from nisl.io import NiftiMasker
dataset = datasets.fetch_haxby()

### Preprocess data ###########################################################

import numpy as np

# XXX is there a better way to separate data ?
session_target = zip(*np.genfromtxt(dataset.session_target[0], dtype=None,
                                    skiprows=1))
labels = np.asarray(session_target[0])
index, target = np.unique(labels, return_inverse=True)
session = np.asarray(session_target[1])

nifti_masker = NiftiMasker(mask=dataset.mask_vt[0], sessions=session,
                           detrend=True)
nifti_masker.fit(dataset.func[0])
fmri_masked = nifti_masker.transform(dataset.func[0])

no_rest_indices = (labels != 'rest')
target = target[no_rest_indices]
fmri_masked = fmri_masked[no_rest_indices]
session = session[no_rest_indices]

### Prediction function #######################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel and C=1
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=1.)

svc.fit(fmri_masked, target)
y_pred = svc.predict(fmri_masked)

### Visualisation #############################################################

### Look at the discriminating weights
sv = svc.support_vectors_
# reverse masking
niimg = nifti_masker.inverse_transform(sv[0])

# We use a masked array so that the voxels at '-1' are displayed
# transparently
act = np.ma.masked_array(niimg.get_data(), niimg.get_data() == 0)

### Create the figure
import pylab as pl
pl.axis('off')
pl.title('SVM vectors')
#pl.imshow(np.rot90(mean_img[..., 27]), cmap=pl.cm.gray,
#          interpolation='nearest')
pl.imshow(np.rot90(act[..., 27]), cmap=pl.cm.hot,
          interpolation='nearest')
pl.show()

### Cross validation ##########################################################

from sklearn.cross_validation import LeaveOneLabelOut

### Define the cross-validation scheme used for validation.
# Here we use a LeaveOneLabelOut cross-validation on the session, which
# corresponds to a leave-one-session-out
cv = LeaveOneLabelOut(session)

### Compute the prediction accuracy for the different folds (i.e. session)

cv_scores = []
for train, test in cv:
    y_pred = svc.fit(fmri_masked[train], target[train]) \
        .predict(fmri_masked[test])
    cv_scores.append(
        np.sum(y_pred == target[test]) / float(np.size(target[test])))

### Print results #############################################################

### Return the corresponding mean prediction accuracy
classification_accuracy = np.mean(cv_scores)

### Printing the results
print "=== ANOVA ==="
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / np.unique(target).size)
# Classification accuracy: 0.986111  / Chance level: 0.500000
