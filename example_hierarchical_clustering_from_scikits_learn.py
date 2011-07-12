"""
===============================================================
Supervised clustering example, based on a scikits.learn example
===============================================================

This example is based on the scikits.learn one :
Feature agglomeration vs. univariate selection

It performs feature agglomeration on Ward tree
"""

print __doc__

import hierarchical_clustering

import numpy as np
import pylab as pl
from scipy import linalg, ndimage

from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.linear_model import BayesianRidge

###############################################################################
# Generate data
n_iterations = 25
n_samples = 200
size = 40 # image size
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
# Hierarchical clustering
#clf = RidgeCV(alphas=[0.1, 1.0, 10.0])
clf = BayesianRidge()
A = grid_to_graph(n_x=size, n_y=size)
hc = hierarchical_clustering.HierarchicalClustering(clf, A)
tab = hc.fit(X, y, n_iterations, verbose=1)

# For visualising only the clusters of the parcellation
#tab = tab.reshape((size, size))
#pl.imshow(tab, interpolation='nearest')
#pl.show()


#For visualising coeffs of clf :
tab2 = tab.copy()
coefs=hc.coef_
for i in np.unique(tab):
    tab2[tab==i]=coefs[i-1]
tab2 = tab2.reshape((size, size))
pl.close('all')
pl.figure(figsize=(15, 15))
# Plotting true weights
pl.subplot(2, 2, 1)
pl.imshow(coef, interpolation="nearest", cmap=pl.cm.RdBu_r)
pl.title("True weights")
# Plotting the result of the hierarchical clustering
pl.subplot(2, 2, 2)
pl.imshow(tab2, interpolation='nearest', cmap=pl.cm.RdBu_r)
pl.title("Hierarchical Clustering, \n%d iterations,\n %d parcels " % 
        (n_iterations, len(np.unique(tab))))
# Plotting the score of the best parcellation at each iteration
pl.subplot(2, 1, 2)
pl.bar(np.arange(len(hc.scores)), hc.scores)
pl.xlabel('iteration')
pl.ylabel('score of the best parcellation of the iteration')
pl.show()
