from datasets import fetch_star_plus_data
import numpy as np
from scikits.learn.feature_extraction.image import grid_to_graph
from supervised_clustering import SupervisedClusteringClassifier
from scikits.learn.cross_val import KFold, cross_val_score

# Loading data
file_X, file_y = fetch_star_plus_data()
X = np.load(file_X[0])
y = np.load(file_y[0])
mask = X[0, :]
mask[mask!=0] = 1
mask.astype(np.bool)
img_shape = mask.shape
X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))

stop
# Connectivity matrix
print "computing connectivity matrix"
A =  grid_to_graph(n_x=img_shape[0], n_y=img_shape[1], n_z=img_shape[2],
        mask=mask)
sc = SupervisedClusteringClassifier(n_jobs=4, n_iterations=30,
                verbose=1)
cv = KFold(X.shape[0], 4)
print "computing score"
cv_scores = cross_val_score(sc, X, y, cv=cv, n_jobs=1,
                                    verbose=1, iid=True)
print "score : ", np.mean(cv_scores)
