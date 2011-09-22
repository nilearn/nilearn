"""
===============================================================
Supervised clustering example, based on a scikits.learn example
===============================================================

A script to compare the performances of Hierarchical Clustering 
and a simple estimator

WARNING : can be a bit long...
"""

print __doc__

import hierarchical_clustering

import numpy as np
import pylab as pl
from scipy import linalg, ndimage

from scikits.learn.cross_val import KFold
from scikits.learn.externals.joblib import Parallel, delayed
from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.linear_model import BayesianRidge

###############################################################################
# Generate data
n_folds = 10 # for cross_validation of final result
n_iterations = 50
n_samples = 200
size = 40
roi_size = 15
snr = 5.
np.random.seed(0)
mask = np.ones([size, size], dtype=np.bool)

coef = np.zeros((size, size))
coef[0:roi_size, 0:roi_size] = -1.
coef[-roi_size:, -roi_size:] = 1.

X = np.random.randn(n_samples, size**2)
for x in X: # smooth data
    x[:] = ndimage.gaussian_filter(x.reshape(size, size), sigma=1.0).ravel()
X -= X.mean(axis=0)
X /= X.std(axis=0)

y = np.dot(X, coef.ravel())
noise = np.random.randn(y.shape[0])
noise_coef = (linalg.norm(y, 2) / np.exp(snr / 20.)) / linalg.norm(noise, 2)
y += noise_coef * noise # add noise


###############################################################################
# Cross_validation
def _cross_val_score(estimator, X, y, train, test):
    estimator.fit(X[train], y[train])
    return estimator.score(X[test], y[test])

def cross_val_scores(estimator, X, y):
    cv = KFold(n_samples, k=n_folds)
    scores = Parallel(n_jobs=-1)(
            delayed(_cross_val_score)(estimator, X, y, train, test)
            for train, test in cv)
    return np.array(scores)

###############################################################################
# Hierarchical clustering
clf = BayesianRidge(compute_score=True, verbose=False)
A = grid_to_graph(n_x=size, n_y=size)
hc = hierarchical_clustering.HierarchicalClustering(clf, A)


# Computing an array of the fitted coefs
tab = hc.fit(X, y, n_iterations, verbose=1)
tab2 = tab.copy()
coefs=hc.coef_
for i in np.unique(tab):
    tab2[tab==i]=coefs[...,i-1]
tab2 = tab2.reshape((size, size))
## Computing scores
s1 = cross_val_scores(estimator=hc, X=X, y=y)
s1 = np.mean(s1)

# Results with just a simple estimator
## Computing score
s2 = cross_val_scores(estimator=clf, X=X, y=y)
s2 = np.mean(s2)
clf.fit(X, y)

# Plotting the result
pl.close('all')
pl.figure(figsize=(15, 15))
# Plotting true weights
pl.subplot(2, 3, 1)
pl.imshow(coef, interpolation="nearest", cmap=pl.cm.RdBu_r)
pl.title("True weights")
# Plotting the result of the hierarchical clustering
# (if you want to plot the clusters, not the coefs, 
# tab = tab.reshape((size, size)), then plot tab
pl.subplot(2, 3, 2)
pl.imshow(tab2, interpolation='nearest', cmap=pl.cm.RdBu_r)
pl.title("Hierarchical Clustering, \n%d iterations,\n %d parcels " % 
        (n_iterations, len(np.unique(tab))))
#Plotting the result of a simple Ridge Regression
pl.subplot(2, 3, 3)
pl.imshow(clf.coef_.reshape((size, size)), interpolation='nearest', cmap=pl.cm.RdBu_r)
pl.title("Simple Bayesian Ridge")
# Plotting the score of the best parcellation at each iteration
pl.subplot(2, 2, 3)
pl.bar(np.arange(len(hc.scores)), hc.scores)
pl.xlabel('iteration of the hierarchical clustering')
pl.ylabel('score of the best parcellation of the iteration')
# Plotting the scores of the final solutions
pl.subplot(2, 2, 4)
pl.bar([0, 1], [s1, s2])
pl.xlabel('Hierarchical_Clustering  vs   SimpleEstimator')
pl.ylabel('Mean of the cross-validated scores')
pl.title('Scores by obtained by cross_validation\n%d folds' % n_folds)
pl.show()
