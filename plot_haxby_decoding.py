"""
The haxby dataset: face vs house in object recognition
=======================================================

A significant part of the running time of this example is actually spent
in loading the data: we load all the data but only use the face and
houses conditions.
"""

### Load Haxby dataset ########################################################
from nisl import datasets
dataset = datasets.fetch_haxby()
fmri_data = dataset.data
mask = dataset.mask
affine = dataset.affine
y = dataset.target
conditions = dataset.target_strings
session = dataset.session

# fmri_data.shape is (40, 64, 64, 1452)
# and mask.shape is (40, 64, 64)

### Preprocess data ###########################################################
import numpy as np

# Build the mean image because we have no anatomic data
mean_img = fmri_data.mean(axis=-1)

# Process the data in order to have a two-dimensional design matrix X of
# shape (n_samples, n_features).
X = fmri_data[mask].T

# X.shape is (n_samples, n_features): (1452, 39912)

# Detrend data on each session independently
from scipy import signal
for s in np.unique(session):
    X[session == s] = signal.detrend(X[session == s], axis=0)


### Restrict to faces and houses ##############################################

# Keep only data corresponding to face or houses
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = X[condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

# We now have n_samples, n_features = X.shape = 864, 39912
n_samples, n_features = X.shape

# We have 2 conditions
n_conditions = np.size(np.unique(y))

### Prediction function #######################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel and C=1
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.)

### Dimension reduction #######################################################

from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 1000
feature_selection = SelectKBest(f_classif, k=1000)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

### Fit and predict ###########################################################

anova_svc.fit(X, y)
y_pred = anova_svc.predict(X)

### Visualisation #############################################################


### Look at the discriminating weights
svc = clf.support_vectors_
# reverse feature selection
svc = feature_selection.inverse_transform(svc)
# reverse masking
act = np.zeros(mean_img.shape)
act[mask != 0] = svc[0]

# We use a masked array so that the voxels at '0' are displayed
# transparently
act = np.ma.masked_array(act, act == 0)

### Create the figure
import pylab as pl
pl.axis('off')
pl.title('SVM vectors')
pl.imshow(np.rot90(mean_img[..., 27]), cmap=pl.cm.gray,
          interpolation='nearest')
pl.imshow(np.rot90(act[..., 27]), cmap=pl.cm.hot,
          interpolation='nearest')
pl.show()

# Saving the results as a Nifti file may also be important
import nibabel
img = nibabel.Nifti1Image(act, affine)
nibabel.save(img, 'haxby_face_vs_house.nii')

### Cross validation ##########################################################

from sklearn.cross_validation import LeaveOneLabelOut

### Define the cross-validation scheme used for validation.
# Here we use a LeaveOneLabelOut cross-validation on the session, which
# corresponds to a leave-one-session-out
cv = LeaveOneLabelOut(session)

### Compute the prediction accuracy for the different folds (i.e. session)
cv_scores = []
for train, test in cv:
    y_pred = anova_svc.fit(X[train], y[train]).predict(X[test])
    cv_scores.append(np.sum(y_pred == y[test]) / float(np.size(y[test])))

### Print results #############################################################

### Return the corresponding mean prediction accuracy
classification_accuracy = np.mean(cv_scores)

### Printing the results
print "=== ANOVA ==="
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / n_conditions)
# Classification accuracy: 0.986111  / Chance level: 0.500000
