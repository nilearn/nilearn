"""
The haxby dataset: face vs house in object recognition
=======================================================

A significant part of the running time of this example is actually spent
in loading the data: we load all the data but only use the face and
houses conditions.
"""

### Load Haxby dataset ########################################################
from nisl import datasets
import numpy as np
import nibabel
dataset_files = datasets.fetch_haxby()

# fmri_data and mask are copied to lose the reference to the original data
bold_img = nibabel.load(dataset_files['data'])
fmri_data = np.copy(bold_img.get_data())
affine = bold_img.get_affine()
y, session = np.loadtxt(dataset_files['session_target']).astype("int").T
conditions = np.recfromtxt(dataset_files['conditions_target'])['f0']
mask = dataset_files['mask']
# fmri_data.shape is (40, 64, 64, 1452)
# and mask.shape is (40, 64, 64)

### Preprocess data ###########################################################
# Build the mean image because we have no anatomic data
mean_img = fmri_data.mean(axis=-1)

### Restrict to faces and houses ##############################################
from nibabel import Nifti1Image

# Keep only data corresponding to face or houses
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = fmri_data[..., condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

# We have 2 conditions
n_conditions = np.size(np.unique(y))

### Loading step ##############################################################
from nisl import mri_transformer
# Detrending is disabled as we are not yet able to do it by session
mri_loader = mri_transformer.MRITransformer(mask=mask, detrend=True,
        copy=False, sessions=session)

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
anova_svc = Pipeline([('load', mri_loader), ('anova', feature_selection),
    ('svc', clf)])

### Fit and predict ###########################################################

anova_svc.fit(Nifti1Image(X, affine), y)
y_pred = anova_svc.predict(Nifti1Image(X, affine))

### Visualisation #############################################################


### Look at the discriminating weights
svc = clf.support_vectors_
# reverse feature selection
svc = feature_selection.inverse_transform(svc)
# reverse masking
niimg = mri_loader.inverse_transform(svc[0])

# We use a masked array so that the voxels at '-1' are displayed
# transparently
act = np.ma.masked_array(niimg.get_data(), niimg.get_data() == 0)

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
    y_pred = anova_svc.fit(Nifti1Image(X[..., train], affine), y[train]) \
        .predict(Nifti1Image(X[..., test], affine))
    cv_scores.append(np.sum(y_pred == y[test]) / float(np.size(y[test])))

### Print results #############################################################

### Return the corresponding mean prediction accuracy
classification_accuracy = np.mean(cv_scores)

### Printing the results
print "=== ANOVA ==="
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / n_conditions)
# Classification accuracy: 0.986111  / Chance level: 0.500000
