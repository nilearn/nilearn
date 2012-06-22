"""Module to lauch the supervised clustering on star plus data
"""

from datasets import fetch_star_plus
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from supervised_clustering import SupervisedClusteringClassifier
from sklearn.cross_val import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
import pylab as pl

# Loading data
data = fetch_star_plus()
data = data[0]
X = data.data
y = data.target
mask = data.mask
img_shape = mask.shape
X = X[:, mask!=0]

# Binarizing y
y = y.astype(np.bool)

# Connectivity matrix
print "computing connectivity matrix"
A =  grid_to_graph(n_x=img_shape[0], n_y=img_shape[1], n_z=img_shape[2],
        mask=mask)
estimator = SVC(kernel='linear', C=1.)
sc = SupervisedClusteringClassifier(estimator=estimator, n_jobs=1, n_iterations=30,
                verbose=1, cv=5, connectivity=A)
cv = StratifiedKFold(y, 5)
print "computing score"
cv_scores = cross_val_score(sc, X, y, cv=cv, n_jobs=8, verbose=1)
sc.fit(X, y)
print "classification score : ", np.mean(cv_scores)
print "number of parcels : %d" % len(np.unique(sc.labels_))

coefs = np.zeros(img_shape)
coefs[mask!=0] = sc.inverse_transform()
pl.figure()
vminmax = np.max(np.abs(coefs))
vmin = -vminmax
vmax = +vminmax
for i in range(8):
    pl.subplot(2, 4, i+1)
    pl.contour(mask[:, :, i])
    pl.imshow(coefs[:, :, i], interpolation='nearest',
            vmin=vmin, vmax=vmax, cmap=pl.cm.RdBu_r)
pl.title('supervised clustering on star plus\n\
        using classification, 300 iteration')
