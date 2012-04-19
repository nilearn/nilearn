### All the imports
import numpy as np
from scipy import signal
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
from nisl import datasets
import nibabel as ni
import os
import pylab

### Load data
data = datasets.fetch_haxby_data()
y = data.target
session = data.session
X = data.data
mask = data.mask
img_shape = X[..., 0].shape
original_img = X[..., 0]
mean_img = X.mean(-1)

# Process the data in order to have a two-dimensional design matrix X of
# shape (nb_samples, nb_features).
X = X[mask!=0].T

print "detrending data"
# Detrend data on each session independently
for s in np.unique(session):
    X[session==s] = signal.detrend(X[session==s], axis=0)

print "removing mask"
# Remove volumes corresponding to rest
X, y, session = X[y!=0], y[y!=0], session[y!=0]
n_samples, n_features = X.shape
n_conditions = np.size(np.unique(y))

X = X[y<=2]
session = session[y<=2]
y = y[y<=2]
session /= 5

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel and C=1
clf = SVC(kernel='linear', C=1.)

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)

### We combine the dimension reduction and the prediction function
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

### Define the cross-validation scheme used for validation.
# Here we use a LeaveOneLabelOut cross-validation on the session, which
# corresponds to a leave-one-session-out
cv = LeaveOneLabelOut(session)

### Compute the prediction accuracy for the different folds (i.e. session)
cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=-1, verbose=1)

### Return the corresponding mean prediction accuracy
classification_accuracy = np.mean(cv_scores)

### Printing the results
print "=== ANOVA ==="
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / n_conditions)

#Look at the discriminating weights
# re-train classifier on the full dataset
anova_svc.fit(X, y)
support = anova_svc.steps[0][1].get_support()
coef_ = np.zeros(support.size)
coef_[support] = anova_svc.steps[1][1].coef_
coef = np.zeros(mask.shape)
coef[mask != 0] = coef_

from nipy.labs import viz
path = os.path.join('pymvpa-exampledata', 'mask.nii.gz')
affine = ni.load(os.path.join(
    datasets.get_dataset_dir("haxby2001"), path)).get_affine()
viz.plot_map(coef,
        affine = affine,
        threshold=np.abs(coef).max()/10,
        cmap=viz.cm.cold_hot,
        anat=mean_img,
        anat_affine=affine)

pylab.show()

