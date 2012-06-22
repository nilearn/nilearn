"""
An example of playing with fMRI data to predict the behavior of a subject from
functional images, using searchlight.
This data was originally collected by Marcel Just and his colleagues in Carnegi
Mellon University's CCBI.

It consists in 54 images of around 5000 voxels.
There is 4 possibles labels:
  0 = images should be ignored.
  1 = rest of fixation interval.
  2 = sentence/picture trial, sentence is not negated.
  3 = sentence/picture trial, sentence is negated.

And example of analysis can be found here:
https://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/theo-73/www/papers/XueruiReport-10-2002.pdf
Additional info on the data can be found here:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/README-data-
documentation.txt

Author: Vincent MICHEL vincent.michel@inria.fr
        Alexandre Gramfort alexandre.gramfort@inria.fr
License: BSD
"""
import numpy as np
from scipy import sparse

from sklearn import neighbors
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import precision_score
from sklearn import svm

from nisl import searchlight, datasets

starplus = datasets.fetch_star_plus(indices=True)
mask = starplus[0].mask.astype(np.int32)
X = starplus[0].data
y = starplus[0].target

### Create the adjacency matrix
clf = neighbors.NearestNeighbors(radius=4)
dist, ind = clf.fit(mask).kneighbors(mask)
A = sparse.lil_matrix((mask.shape[0], mask.shape[0]))
for i, li in enumerate(ind):
    A[i, list(li[1:])] = np.ones(len(li[1:]))

### Instanciate the searchlight model
n_jobs = 2
estimator = svm.SVC(kernel='linear', C=1)
searchlight = searchlight.SearchLight(A, estimator, n_jobs=n_jobs)
score_func = precision_score
cv = LeaveOneOut(n=2)
scores = searchlight.fit(X, y, score_func=score_func, cv=cv)
