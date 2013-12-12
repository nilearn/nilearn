"""
The haxby dataset: face vs house in object recognition
=======================================================

A significant part of the running time of this example is actually spent
in loading the data: we load all the data but only use the face and
houses conditions.
"""

### Load Haxby dataset ########################################################
from nilearn import datasets
import numpy as np
import nibabel
dataset_files = datasets.fetch_haxby_simple()

# fmri_data and mask are copied to break any reference to the original object
bold_img = nibabel.load(dataset_files.func)
fmri_data = np.copy(bold_img.get_data())
affine = bold_img.get_affine()
y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
mask = dataset_files.mask
# fmri_data.shape is (40, 64, 64, 1452)
# and mask.shape is (40, 64, 64)

### Preprocess data ###########################################################
# Build the mean image because we have no anatomic data
mean_img = fmri_data.mean(axis=-1)

### Restrict to faces and houses ##############################################

# Keep only data corresponding to faces or houses
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = fmri_data[..., condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

# We have 2 conditions
n_conditions = np.size(np.unique(y))

### Loading step ##############################################################
from nilearn.input_data import NiftiMasker
from nibabel import Nifti1Image
nifti_masker = NiftiMasker(mask=mask, sessions=session, smoothing_fwhm=4,
                           memory="nilearn_cache", memory_level=1)
niimg = Nifti1Image(X, affine)
X = nifti_masker.fit_transform(niimg)

### Prediction function #######################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVC
svc = SVC(kernel='linear')

### Dimension reduction #######################################################

from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

### Fit and predict ###########################################################

anova_svc.fit(X, y)
y_pred = anova_svc.predict(X)

### Visualisation #############################################################

### Look at the SVC's discriminating weights
coef = svc.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)
# reverse masking
weight_niimg = nifti_masker.inverse_transform(coef)

# We use a masked array so that the voxels at '-1' are displayed
# transparently
weights = np.ma.masked_array(weight_niimg.get_data(),
                             weight_niimg.get_data() == 0)

### Create the figure
import matplotlib.pyplot as plt
plt.figure(figsize=(3, 5))
plt.imshow(np.rot90(mean_img[..., 27]), cmap=plt.cm.gray,
          interpolation='nearest')
plt.imshow(np.rot90(weights[..., 27, 0]), cmap=plt.cm.hot,
          interpolation='nearest')
plt.axis('off')
plt.title('SVM weights')
plt.show()

### Saving the results as a Nifti file may also be important
import nibabel
nibabel.save(weight_niimg, 'haxby_face_vs_house.nii')

### Cross validation ##########################################################

from sklearn.cross_validation import LeaveOneLabelOut

### Define the cross-validation scheme used for validation.
# Here we use a LeaveOneLabelOut cross-validation on the session label
# divided by 2, which corresponds to a leave-two-session-out
cv = LeaveOneLabelOut(session // 2)

### Compute the prediction accuracy for the different folds (i.e. session)
cv_scores = []
for train, test in cv:
    anova_svc.fit(X[train], y[train])
    y_pred = anova_svc.predict(X[test])
    cv_scores.append(np.sum(y_pred == y[test]) / float(np.size(y[test])))

### Print results #############################################################

### Return the corresponding mean prediction accuracy
classification_accuracy = np.mean(cv_scores)

### Printing the results
print "=== ANOVA ==="
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / n_conditions)
# Classification accuracy: 0.986111  / Chance level: 0.500000
