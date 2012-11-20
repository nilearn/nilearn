"""
Test the searchlight module
"""
# Author: Alexandre Abraham
# License: simplified BSD

from nose.tools import assert_equal
import numpy as np

# Create a toy dataset to run searchlight on

# Initialize with 4x4x4 scans of random values on 30 frames
rand = np.random.RandomState(0)
frames = 30
data = rand.rand(frames, 5, 5, 5)
mask = np.ones((5, 5, 5), np.bool)

# Create a condition array
cond = np.array([int(i > (frames / 2)) for i in range(frames)])

# Create an activation pixel.
data[:, 2, 2, 2] = 0
data.T[2, 2, 2][cond.astype(np.bool)] = 2

# Define score function
from sklearn.metrics import precision_score
score_func = precision_score

# Define cross validation
from sklearn.cross_validation import KFold
cv = KFold(cond.size, k=4)

from nisl import searchlight
n_jobs = 1

# Run Searchlight with different radii
# Small radius : only one pixel is selected
sl = searchlight.SearchLight(mask, mask, radius=0.5,
                             n_jobs=n_jobs, score_func=score_func, cv=cv)
sl.fit(data, cond)
assert_equal(np.where(sl.scores_ == 1)[0].size, 1)
assert_equal(sl.scores_[2, 2, 2], 1.)

# Medium radius : little ball selected

sl = searchlight.SearchLight(mask, mask, radius=1,
                             n_jobs=n_jobs, score_func=score_func, cv=cv)
sl.fit(data, cond)
assert_equal(np.where(sl.scores_ == 1)[0].size, 7)
assert_equal(sl.scores_[2, 2, 2], 1.)
assert_equal(sl.scores_[1, 2, 2], 1.)
assert_equal(sl.scores_[2, 1, 2], 1.)
assert_equal(sl.scores_[2, 2, 1], 1.)
assert_equal(sl.scores_[3, 2, 2], 1.)
assert_equal(sl.scores_[2, 3, 2], 1.)
assert_equal(sl.scores_[2, 2, 3], 1.)

# Big radius : big ball selected
sl = searchlight.SearchLight(mask, mask, radius=2,
                             n_jobs=n_jobs, score_func=score_func, cv=cv)
sl.fit(data, cond)
assert_equal(np.where(sl.scores_ == 1)[0].size, 33)
assert_equal(sl.scores_[2, 2, 2], 1.)
