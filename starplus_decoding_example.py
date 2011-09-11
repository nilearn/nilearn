""" 
================= Supervised Clustering =====================
Decoding example using supervised clustering on starplus data
"""

print __doc__

import numpy as np
from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.cross_val import StratifiedKFold, cross_val_score
from scikits.learn.svm import SVC

from datasets import fetch_star_plus_data
from supervised_clustering import SupervisedClusteringClassifier


# Loading data
data = fetch_star_plus_data()
scores = []

# We compute the score for each patient
for i in range(6):
    # Using the data corresponding to the patient
    X = data.datas[i]
    y = data.targets[i]
    mask = data.masks[i]
    img_shape = mask.shape
    X = X[:, mask!=0]

    # Binarizing y to perform classification
    y = y.astype(np.bool)

    # Computing connectivity matrix
    A =  grid_to_graph(n_x=img_shape[0], n_y=img_shape[1], n_z=img_shape[2],
            mask=mask)
    estimator = SVC(kernel='linear', C=1.)
    sc = SupervisedClusteringClassifier(estimator=estimator, n_jobs=1,
            n_iterations=150, cv=6, connectivity=A, verbose=0)
    cv = StratifiedKFold(y, 10)
    print "Computing score for the patient %d on 6" % i
    cv_scores = cross_val_score(sc, X, y, cv=cv, n_jobs=8, verbose=0)
    sc.fit(X, y)
    print ". Classification score for patient %d : %f" % (i, np.mean(cv_scores))
    print ". Number of parcels : %d" % len(np.unique(sc.labels_))
    scores.append(np.mean(cv_scores))


print "===================================="
print "Average score for the whole dataset : %f", np.mean(scores)
