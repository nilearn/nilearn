"""Test the searchlight module."""
# Author: Alexandre Abraham
# License: simplified BSD

import nibabel
import numpy as np
from nilearn.decoding import searchlight
from sklearn.model_selection import KFold


def _make_searchlight_test_data(frames):
    # Initialize with 4x4x4 scans of random values on 30 frames
    rand = np.random.RandomState(0)
    frames = frames
    data = rand.rand(5, 5, 5, frames)
    mask = np.ones((5, 5, 5), dtype=bool)
    mask_img = nibabel.Nifti1Image(mask.astype("uint8"), np.eye(4))
    # Create a condition array, with balanced classes
    cond = np.arange(frames, dtype=int) >= (frames // 2)

    # Create an activation pixel.
    data[2, 2, 2, :] = 0
    data[2, 2, 2][cond.astype(bool)] = 2
    data_img = nibabel.Nifti1Image(data, np.eye(4))

    return data_img, cond, mask_img


def define_cross_validation():
    # Define cross validation
    cv = KFold(n_splits=4)
    n_jobs = 1
    return cv, n_jobs


def test_searchlight_small_radius():
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()

    # Small radius : only one pixel is selected
    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=mask_img,
        radius=0.5,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=cv,
        verbose=1,
    )
    sl.fit(data_img, cond)

    assert np.where(sl.scores_ == 1)[0].size == 1
    assert sl.scores_[2, 2, 2] == 1.0


def test_searchlight_mask_far_from_signal():
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()

    process_mask = np.zeros((5, 5, 5), dtype=bool)
    process_mask[0, 0, 0] = True
    process_mask_img = nibabel.Nifti1Image(
        process_mask.astype("uint8"), np.eye(4)
    )
    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=process_mask_img,
        radius=0.5,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=cv,
    )
    sl.fit(data_img, cond)

    assert np.where(sl.scores_ == 1)[0].size == 0


def test_searchlight_medium_radius():
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()

    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=mask_img,
        radius=1,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=cv,
    )
    sl.fit(data_img, cond)

    assert np.where(sl.scores_ == 1)[0].size == 7
    assert sl.scores_[2, 2, 2] == 1.0
    assert sl.scores_[1, 2, 2] == 1.0
    assert sl.scores_[2, 1, 2] == 1.0
    assert sl.scores_[2, 2, 1] == 1.0
    assert sl.scores_[3, 2, 2] == 1.0
    assert sl.scores_[2, 3, 2] == 1.0
    assert sl.scores_[2, 2, 3] == 1.0


def test_searchlight_large_radius():
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()

    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=mask_img,
        radius=2,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=cv,
    )
    sl.fit(data_img, cond)
    assert np.where(sl.scores_ == 1)[0].size == 33
    assert sl.scores_[2, 2, 2] == 1.0


def group_cross_validation(cv):
    try:
        from sklearn.model_selection import LeaveOneGroupOut

        gcv = LeaveOneGroupOut()
    except ImportError:
        # won't import model selection if it's not there.
        # the groups variable should have no effect.
        gcv = cv
    return gcv


def test_searchlight_group_cross_validation():
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()
    gcv = group_cross_validation(cv)

    groups = np.random.RandomState(42).permutation(
        np.arange(frames, dtype=int) > (frames // 2)
    )

    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=mask_img,
        radius=1,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=gcv,
    )
    sl.fit(data_img, cond, groups)
    assert np.where(sl.scores_ == 1)[0].size == 7
    assert sl.scores_[2, 2, 2] == 1.0


def test_searchlight_group_cross_validation_with_extra_group_variable():
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()

    groups = np.random.RandomState(42).permutation(
        np.arange(frames, dtype=int) > (frames // 2)
    )

    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=mask_img,
        radius=1,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=cv,
    )
    sl.fit(data_img, cond, groups)
    assert np.where(sl.scores_ == 1)[0].size == 7
    assert sl.scores_[2, 2, 2] == 1.0

    # Check whether searchlight works on list of 3D images
    rand = np.random.RandomState(0)
    data = rand.rand(5, 5, 5)
    data_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    imgs = [data_img] * 12

    # labels
    y = [0, 1] * 6

    # run searchlight on list of 3D images
    sl = searchlight.SearchLight(mask_img)
    sl.fit(imgs, y)
