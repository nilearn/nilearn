### All the imports
import numpy as np
from scipy import signal
import nibabel as ni
from scikits.learn.svm import SVC
from scikits.learn.feature_selection import SelectKBest, f_classif
from scikits.learn.pipeline import Pipeline
from scikits.learn.cross_val import LeaveOneLabelOut, cross_val_score

### Load data
y, session = np.loadtxt("attributes.txt").astype("int").T
X = ni.load("bold.nii.gz").get_data()
mask = ni.load("mask.nii.gz").get_data()

# Process the data in order to have a two-dimensional design matrix X of
# shape (nb_samples, nb_features).
X = X[mask!=0].T

# Detrend data on each session independently
for s in np.unique(session):
    X[session==s] = signal.detrend(X[session==s], axis=0)

# Remove volumes corresponding to rest
X, y, session = X[y!=0], y[y!=0], session[y!=0]
n_samples, n_features = X.shape
n_conditions = np.size(np.unique(y))

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
cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=-1,
                            verbose=1, iid=True)

### Return the corresponding mean prediction accuracy
classification_accuracy = np.sum(cv_scores) / float(n_samples)
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / n_conditions)

