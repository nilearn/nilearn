### All the imports
import numpy as np
from scipy import signal
import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.cross_val import LeaveOneLabelOut, cross_val_score

### Load data
data = datasets.fetch_haxby_data()
y = data.target
session = data.session
X = data.data
mask = data.mask
img_shape = X[..., 0].shape
original_img = X[..., 0]

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

#### Same test using the supervised clustering
#from sklearn.feature_extraction.image import grid_to_graph
#from supervised_clustering import SupervisedClusteringClassifier
#estimator = SVC(kernel='linear', C=1.)
#A =  grid_to_graph(n_x=img_shape[0], n_y=img_shape[1], n_z=img_shape[2], mask=mask)
#print "computed connectivity matrix"
#sc = SupervisedClusteringClassifier(estimator=estimator, connectivity=A, n_jobs=1,
#        cv=5, n_iterations=50, verbose=1)
#cv_scores = cross_val_score(sc, X, y, cv=cv, n_jobs=4, verbose=1)
#
#sc.fit(X, y)
#computed_coefs = sc.inverse_transform()

### Printing the results
print "=== ANOVA ==="
print "Classification accuracy: %f" % classification_accuracy, \
    " / Chance level: %f" % (1. / n_conditions)

#classification_accuracy = np.mean(cv_scores)
#print "=== SUPERVISED CLUSTERING ==="
#print "Classification accuracy: %f" % classification_accuracy, \
#    " / Chance level: %f" % (1. / n_conditions)
#print "Number of parcellations : %d" % len(np.unique(sc.labels_))
#
################################################################################
## Ploting the results
#import pylab as pl
#pl.close('all')
#pl.figure()
#pl.title('Scores of the supervised clustering')
#pl.subplot(2, 1, 1)
#pl.bar(np.arange(len(sc.scores_)), sc.scores_)
#pl.xlabel('scores')
#pl.ylabel('iteration')
#pl.title('Score of the best parcellation of each iteration')
#pl.subplot(2, 1, 2)
#pl.bar(np.arange(len(sc.delta_scores_)), sc.delta_scores_)
#pl.xlabel('delta_scores (min = %f) ' % sc.score_min_)
#pl.ylabel('iteration')
#
#coef_ = np.zeros(mask.shape)
#coef_[mask!=0] = computed_coefs
#coef_ = coef_.reshape(img_shape)
#
#pl.figure()
#pl.subplot(2, 1, 1)
#pl.title('Original image')
#pl.contour(mask[:, :, img_shape[2]/2])
#pl.imshow(original_img[:, :, img_shape[2]/2])
#pl.subplot(2, 1, 2)
#pl.title('cut at z/2')
#vminmax = np.max(np.abs(computed_coefs))
#vmin = -vminmax
#vmax = +vminmax
#pl.contour(mask[:, :, img_shape[2]/2])
#pl.imshow(coef_[:, :, img_shape[2]/2], interpolation='nearest',
#        vmin=vmin, vmax=vmax, cmap=pl.cm.RdBu_r)
#
#pl.show()
