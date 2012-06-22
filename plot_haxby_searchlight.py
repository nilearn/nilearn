import numpy as np
from scipy import sparse, signal

from sklearn import neighbors
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score

from nisl import searchlight, datasets

### Download the data from the web
dataset = datasets.fetch_haxby()

X = dataset.data
mask = dataset.mask
y = dataset.target
session = dataset.session
X = X[mask != 0].T

mask = np.asarray(np.where(mask)).T
print "detrending data"
for s in np.unique(session):
    X[session == s] = signal.detrend(X[session == s], axis=0)

# Remove volumes corresponding to rest
X, y, session = X[y != 0], y[y != 0], session[y != 0]
X, y, session = X[y < 3], y[y < 3], session[y < 3]

### Create the adjacency matrix
clf = neighbors.NearestNeighbors(radius=4., n_neighbors=50)
dist, ind = clf.fit(mask).kneighbors(mask)
A = sparse.lil_matrix((mask.shape[0], mask.shape[0]))
for i, li in enumerate(ind):
    A[i, list(li[1:])] = np.ones(len(li[1:]))

### Instanciate the searchlight model
n_jobs = 2
searchlight = searchlight.SearchLight(A, n_jobs=n_jobs)
score_func = precision_score
cv = KFold(y.size, k=4)
# cv = None
scores = searchlight.fit(X, y, score_func=score_func, verbose=True, cv=cv)
