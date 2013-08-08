"""
Test the searchlight module
"""
# Author: Alexandre Abraham
# License: simplified BSD

from nose.tools import assert_equal
import numpy as np
import nibabel
from .. import searchlight


def test_searchlight():
    # Create a toy dataset to run searchlight on

    # Initialize with 4x4x4 scans of random values on 30 frames
    rand = np.random.RandomState(0)
    frames = 30
    data = rand.rand(5, 5, 5, frames)
    mask = np.ones((5, 5, 5), np.bool)
    mask_img = nibabel.Nifti1Image(mask.astype(np.int), np.eye(4))
    # Create a condition array
    cond = np.asarray([int(i > (frames / 2)) for i in range(frames)])

    # Create an activation pixel.
    data[2, 2, 2, :] = 0
    data[2, 2, 2][cond.astype(np.bool)] = 2
    data_img = nibabel.Nifti1Image(data, np.eye(4))

    # Define score function
    try:
        from sklearn.metrics import accuracy_score
        score_func = accuracy_score
    except:
        from sklearn.metrics import prediction_score
        score_func = prediction_score

    # Define cross validation
    from sklearn.cross_validation import KFold
    cv = KFold(cond.size, k=4)

    n_jobs = 1

    # Run Searchlight with different radii
    # Small radius : only one pixel is selected
    sl = searchlight.SearchLight(mask_img, process_mask_img=mask_img,
                                 radius=0.5, n_jobs=n_jobs,
                                 score_func=score_func, cv=cv)
    sl.fit(data_img, cond)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 1)
    assert_equal(sl.scores_[2, 2, 2], 1.)

    # Medium radius : little ball selected

    sl = searchlight.SearchLight(mask_img, process_mask_img=mask_img, radius=1,
                                 n_jobs=n_jobs, score_func=score_func, cv=cv)
    sl.fit(data_img, cond)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 7)
    assert_equal(sl.scores_[2, 2, 2], 1.)
    assert_equal(sl.scores_[1, 2, 2], 1.)
    assert_equal(sl.scores_[2, 1, 2], 1.)
    assert_equal(sl.scores_[2, 2, 1], 1.)
    assert_equal(sl.scores_[3, 2, 2], 1.)
    assert_equal(sl.scores_[2, 3, 2], 1.)
    assert_equal(sl.scores_[2, 2, 3], 1.)

    # Big radius : big ball selected
    sl = searchlight.SearchLight(mask_img, process_mask_img=mask_img, radius=2,
                                 n_jobs=n_jobs, score_func=score_func, cv=cv)
    sl.fit(data_img, cond)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 33)
    assert_equal(sl.scores_[2, 2, 2], 1.)
