"""
Test the searchlight module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import numpy as np
import nibabel
import sklearn
from distutils.version import LooseVersion
from nose.tools import assert_equal
from nilearn.decoding import searchlight


def test_searchlight():
    # Create a toy dataset to run searchlight on

    # Initialize with 4x4x4 scans of random values on 30 frames
    rand = np.random.RandomState(0)
    frames = 30
    data = rand.rand(5, 5, 5, frames)
    mask = np.ones((5, 5, 5), np.bool)
    mask_img = nibabel.Nifti1Image(mask.astype(np.int), np.eye(4))
    # Create a condition array
    cond = np.arange(frames, dtype=int) > (frames // 2)

    # Create an activation pixel.
    data[2, 2, 2, :] = 0
    data[2, 2, 2][cond.astype(np.bool)] = 2
    data_img = nibabel.Nifti1Image(data, np.eye(4))

    # Define cross validation
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=4)
    n_jobs = 1

    # Run Searchlight with different radii
    # Small radius : only one pixel is selected
    sl = searchlight.SearchLight(mask_img, process_mask_img=mask_img,
                                 radius=0.5, n_jobs=n_jobs,
                                 scoring='accuracy', cv=cv)
    sl.fit(data_img, cond)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 1)
    assert_equal(sl.scores_[2, 2, 2], 1.)

    # The voxel selected in process_mask_img is too far from the signal
    process_mask = np.zeros((5, 5, 5), np.bool)
    process_mask[0, 0, 0] = True
    process_mask_img = nibabel.Nifti1Image(process_mask.astype(np.int),
                                           np.eye(4))
    sl = searchlight.SearchLight(mask_img, process_mask_img=process_mask_img,
                                 radius=0.5, n_jobs=n_jobs,
                                 scoring='accuracy', cv=cv)
    sl.fit(data_img, cond)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 0)

    # Medium radius : little ball selected
    sl = searchlight.SearchLight(mask_img, process_mask_img=mask_img, radius=1,
                                 n_jobs=n_jobs, scoring='accuracy', cv=cv)
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
                                 n_jobs=n_jobs, scoring='accuracy', cv=cv)
    sl.fit(data_img, cond)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 33)
    assert_equal(sl.scores_[2, 2, 2], 1.)

    # group cross validation
    try:
        from sklearn.model_selection import LeaveOneGroupOut
        gcv = LeaveOneGroupOut()
    except ImportError:
        # won't import model selection if it's not there.
        # the groups variable should have no effect.
        gcv = cv

    groups = np.random.permutation(np.arange(frames, dtype=int) >
                                   (frames // 2))
    sl = searchlight.SearchLight(mask_img, process_mask_img=mask_img, radius=1,
                                 n_jobs=n_jobs, scoring='accuracy', cv=gcv)
    sl.fit(data_img, cond, groups)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 7)
    assert_equal(sl.scores_[2, 2, 2], 1.)

    # adding superfluous group variable
    sl = searchlight.SearchLight(mask_img, process_mask_img=mask_img, radius=1,
                                 n_jobs=n_jobs, scoring='accuracy', cv=cv)
    sl.fit(data_img, cond, groups)
    assert_equal(np.where(sl.scores_ == 1)[0].size, 7)
    assert_equal(sl.scores_[2, 2, 2], 1.)

    # Check whether searchlight works on list of 3D images
    rand = np.random.RandomState(0)
    data = rand.rand(5, 5, 5)
    data_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    imgs = [data_img, data_img, data_img, data_img, data_img, data_img]

    # labels
    y = [0, 1, 0, 1, 0, 1]

    # run searchlight on list of 3D images
    sl = searchlight.SearchLight(mask_img)
    sl.fit(imgs, y)

