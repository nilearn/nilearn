"""
The haxby dataset: face vs house in object recognition
=======================================================

"""


### Load Haxby dataset ########################################################
from nisl import datasets
dataset = datasets.fetch_haxby_data()
X = dataset.data
mask = dataset.mask
y = dataset.target
session = dataset.session

### Preprocess data ###########################################################
import numpy as np
from scipy import signal

# Build the mean image because we have no anatomic data
mean_img = X.mean(axis=-1)

X.shape
# (40, 64, 64, 1452)
mask.shape
# (40, 64, 64)

# Process the data in order to have a two-dimensional design matrix X of
# shape (nb_samples, nb_features).
X = X[mask != 0].T

X.shape
# (1452, 39912)

# Detrend data on each session independently
print "detrending data"
for s in np.unique(session):
    X[session == s] = signal.detrend(X[session == s], axis=0)


### Remove rest period ########################################################

# Remove volumes corresponding to rest
X, y, session = X[y != 0], y[y != 0], session[y != 0]

# We can check that
n_samples, n_features = X.shape
n_samples
# 864
n_features
# 39912

# Look at target y
y.shape
# (1452,)

# Check conditions:
# - 0 is the rest period
# - [1..8] is the label of each object
np.unique(y)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8])

# We have the 8 conditions
n_conditions = np.size(np.unique(y))
n_conditions
# 8


### Prediction function #######################################################

from sklearn.svm import SVC

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel and C=1
clf = SVC(kernel='linear', C=1.)
clf
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#   kernel='linear', probability=False, scale_C=True, shrinking=True,
#   tol=0.001, verbose=False)

### Dimension reduction #######################################################

from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)
# >>> feature_selection
# SelectKBest(k=500, score_func=<function f_classif at 0x...>)

### Pipeline ##################################################################

from sklearn.pipeline import Pipeline

### We combine the dimension reduction and the prediction function
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
anova_svc
# Pipeline(steps=[('anova', SelectKBest(k=500, score_func=<function
#   f_classif at ...>)), ('svc', SVC(C=1.0, cache_size=200, class_weight=None,
#   coef0=0.0, degree=3, gamma=0.0, kernel='linear', probability=False,
#   scale_C=True, shrinking=True, tol=0.001, verbose=False))])


### Fit and predict ###########################################################

anova_svc.fit(X, y)
# Pipeline(steps=[('anova', SelectKBest(k=500, score_func=<function
#   f_classif at ...>)), ('svc', SVC(C=1.0, cache_size=200, class_weight=None,
#   coef0=0.0, degree=3, gamma=0.0, kernel='linear', probability=False,
#   scale_C=True, shrinking=True, tol=0.001, verbose=False))])

y_pred = anova_svc.predict(X)
y_pred.shape
# (864,)
X.shape
# (864, 39912)

### Visualisation #############################################################

from matplotlib import pyplot as plt

### Look at the discriminating weights
svc = clf.support_vectors_
# reverse feature selection
svc = feature_selection.inverse_transform(svc)
# reverse masking
act = np.zeros(mean_img.shape)
act[mask != 0] = svc[0]
act = np.ma.masked_array(act, act == 0)

### Create the figure on z=20
plt.axis('off')
plt.title('SVM vectors')
plt.imshow(mean_img[:, 20, :], cmap=plt.cm.gray, interpolation=None)
plt.imshow(act[:, 20, :], cmap=plt.cm.hot, interpolation=None)
plt.show()


### Cross validation ##########################################################

from sklearn.cross_validation import LeaveOneLabelOut

### Define the cross-validation scheme used for validation.
# Here we use a LeaveOneLabelOut cross-validation on the session, which
# corresponds to a leave-one-session-out
cv = LeaveOneLabelOut(session)

### Compute the prediction accuracy for the different folds (i.e. session)
#cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=-1, verbose=1)
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
# Classification accuracy: 0.744213  / Chance level: 0.125000
