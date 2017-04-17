"""
An example of playing with fMRI data to predict the behavior of a subject from
functional images, using searchlight.
This data was originally collected by Marcel Just and his colleagues in Carnegie
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
import os
import numpy as np
from scipy import io, sparse
import pylab as pl

from sklearn import neighbors
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import precision_score
from sklearn import svm

from searchlight import SearchLight

### Download the data from the web
if not os.path.exists('data-starplus-04847-v7.mat'):
    # Download the data
    import urllib
    print "Downloading data, Please Wait (79MB)"
    opener = urllib.urlopen('http://www.cs.cmu.edu/afs/cs.cmu.edu/'+
            'project/theo-81/www/data-starplus-04847-v7.mat')
    open('data-starplus-04847-v7.mat', 'wb').write(opener.read())


### Load the data with respect to the Scikit-learn API

mat = io.loadmat('data-starplus-04847-v7.mat', struct_as_record=False)

### Read the mask and meta data
meta_data = mat['meta'][0][0]
n_samples, n_voxels = meta_data.ntrials[0][0], meta_data.nvoxels[0][0]
mask = meta_data.colToCoord

print "Total number of voxels:", n_voxels
print "Total number of samples:", n_samples

### Read raw data
raw_data = mat['data']
X = np.asarray([data[0][0] for data in raw_data])

### Read labels
y = np.ravel([d.cond for d in mat['info'][0]])

### We remove unuseful images (cond 0) and shift the conditions to 0
X = X[y != 0]
y = y[y != 0]

### Here, we will try to classify sentence_picture (2) against
# picture_sentence (3). Rescale y in [0,1].
X = X[y > 1]
y = y[y > 1] - 2
n_samples = y.shape[0]

### Create the adjacency matrix
clf = neighbors.NearestNeighbors(radius=4)
dist, ind = clf.fit(mask).kneighbors(mask)
A = sparse.lil_matrix((mask.shape[0], mask.shape[0]))
for i, li in enumerate(ind):
    A[i, list(li[1:])] = np.ones(len(li[1:]))
    
### Instanciate the searchlight model
n_jobs = 2
estimator = svm.SVC(kernel='linear',C=1)
searchlight = SearchLight(A, estimator, n_jobs=n_jobs)
score_func = precision_score
cv = LeaveOneOut(n=2)
scores = searchlight.fit(X, y, score_func=score_func, cv=cv)
