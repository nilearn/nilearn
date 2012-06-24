import numpy as np
from scipy import signal
from matplotlib import pyplot

from sklearn import neighbors
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score

from nisl import searchlight, datasets

### Fetch the dataset
dataset = datasets.fetch_haxby()

X = dataset.data
y = dataset.target
session = dataset.session

### Prepare the masks
# Here we will use several masks :
# * mask is the original mask
# * process_mask is a subset of mask, it contains voxels that should be
#   processed
mask = (dataset.mask != 0)
process_mask = mask.copy()
process_mask[..., 29:] = False
process_mask[..., :23] = False

X = X[process_mask != 0].T

mask_indices = np.asarray(np.where(mask != 0)).T
process_mask_indices = np.asarray(np.where(process_mask != 0)).T
print "detrending data"
for s in np.unique(session):
    X[session == s] = signal.detrend(X[session == s], axis=0)

# Remove volumes corresponding to rest and keep only face and house scans
X, y, session = X[y != 0], y[y != 0], session[y != 0]
X, y, session = X[y < 3], y[y < 3], session[y < 3]

### Create the adjacency matrix
# A sphere of a given radius centered on each voxel is taken
clf = neighbors.NearestNeighbors(radius=2.)
A = clf.fit(process_mask_indices).radius_neighbors_graph(process_mask_indices)

### Instanciate the searchlight model
# Make processing parallel
n_jobs = 2
score_func = precision_score
# A cross validation method is needed to measure precision of each voxel
cv = KFold(y.size, k=4)
searchlight = searchlight.SearchLight(A, n_jobs=n_jobs,
        score_func=score_func, verbose=True, cv=cv)
# scores.scores is an array containing per voxel precision
scores = searchlight.fit(X, y)

### Unmask the data and display it
S = np.zeros(mask.shape)
S[process_mask] = scores.scores
pyplot.imshow(np.rot90(S[..., 26]), interpolation='nearest',
        cmap=pyplot.cm.spectral)
pyplot.show()
