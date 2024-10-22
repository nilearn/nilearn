"""Test the searchlight module."""

# Author: Alexandre Abraham

import numpy as np
import pytest
from nibabel import Nifti1Image
from sklearn.model_selection import KFold, LeaveOneGroupOut

from nilearn.conftest import _rng
from nilearn.decoding import searchlight


def _make_searchlight_test_data(frames):
    data = _rng().random((5, 5, 5, frames))
    mask = np.ones((5, 5, 5), dtype=bool)
    mask_img = Nifti1Image(mask.astype("uint8"), np.eye(4))
    # Create a condition array, with balanced classes
    cond = np.arange(frames, dtype=int) >= (frames // 2)

    data[2, 2, 2, :] = 0
    data[2, 2, 2, cond] = 2
    data_img = Nifti1Image(data, np.eye(4))

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
    sl.fit(data_img, y=cond)

    assert np.where(sl.scores_ == 1)[0].size == 1
    assert sl.scores_[2, 2, 2] == 1.0


def test_searchlight_mask_far_from_signal(affine_eye):
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()

    process_mask = np.zeros((5, 5, 5), dtype=bool)
    process_mask[0, 0, 0] = True
    process_mask_img = Nifti1Image(process_mask.astype("uint8"), affine_eye)
    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=process_mask_img,
        radius=0.5,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=cv,
    )
    sl.fit(data_img, y=cond)

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


def test_searchlight_group_cross_validation(rng):
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    _, n_jobs = define_cross_validation()

    groups = rng.permutation(np.arange(frames, dtype=int) > (frames // 2))

    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=mask_img,
        radius=1,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=LeaveOneGroupOut(),
    )
    sl.fit(data_img, y=cond, groups=groups)

    assert np.where(sl.scores_ == 1)[0].size == 7
    assert sl.scores_[2, 2, 2] == 1.0


def test_searchlight_group_cross_validation_with_extra_group_variable(
    rng,
    affine_eye,
):
    frames = 30
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    cv, n_jobs = define_cross_validation()

    groups = rng.permutation(np.arange(frames, dtype=int) > (frames // 2))

    sl = searchlight.SearchLight(
        mask_img,
        process_mask_img=mask_img,
        radius=1,
        n_jobs=n_jobs,
        scoring="accuracy",
        cv=cv,
    )
    sl.fit(data_img, y=cond, groups=groups)

    assert np.where(sl.scores_ == 1)[0].size == 7
    assert sl.scores_[2, 2, 2] == 1.0

    # Check whether searchlight works on list of 3D images
    data = rng.random((5, 5, 5))
    data_img = Nifti1Image(data, affine=affine_eye)
    imgs = [data_img] * 12

    # labels
    y = [0, 1] * 6

    # run searchlight on list of 3D images
    sl = searchlight.SearchLight(mask_img)
    sl.fit(imgs, y)


def test_searchlight_attributes_exist_after_fit():
    """Test if attributes `process_mask_` and `masked_scores_`
    exist after fitting using mock data.
    """
    # Use the existing helper function to generate data
    frames = 20
    data_img, cond, mask_img = _make_searchlight_test_data(frames)

    # Instantiate and fit the SearchLight with mock data
    sl = searchlight.SearchLight(mask_img, radius=1.0)
    sl.fit(data_img, y=cond)

    # Check if attributes exist after fitting
    assert hasattr(sl, "process_mask_")
    assert hasattr(sl, "masked_scores_")


def test_searchlight_scores_img_error_before_fit():
    """Test if accessing `scores_img_` raises an error before fitting."""
    # Create mock mask
    frames = 20
    data_img, cond, mask_img = _make_searchlight_test_data(frames)

    # Instantiate SearchLight without fitting
    sl = searchlight.SearchLight(mask_img, radius=5.0)

    # Check if accessing `scores_img_` raises a ValueError
    with pytest.raises(ValueError, match="The model has not been fitted yet."):
        sl.scores_img_


def test_dimension_mismatch_error():
    """Test if ValueError is raised when mask and image dimensions mismatch."""
    data_img, cond, _ = _make_searchlight_test_data(frames=20)

    # Create a mask with mismatched dimensions but valid content
    invalid_mask_img = Nifti1Image(
        np.ones((4, 4, 4), dtype="uint8"), np.eye(4)
    )

    sl = searchlight.SearchLight(invalid_mask_img, radius=1.0)

    with pytest.raises(
        ValueError, match="The mask image and the 4D input images must"
    ):
        sl.fit(data_img, y=cond)




def test_transform_without_fit():
    """Test if calling `transform()` raises ValueError before fitting."""
    frames = 20
    data_img, cond, mask_img = _make_searchlight_test_data(frames)
    sl = searchlight.SearchLight(mask_img, radius=1.0)

    with pytest.raises(ValueError, match="fit the model before calling"):
        sl.transform(data_img)


def test_transform_applies_mask_correctly():
    """Test if `transform()` applies the mask correctly."""
    frames = 20
    data_img, cond, mask_img = _make_searchlight_test_data(frames)

    sl = searchlight.SearchLight(mask_img, radius=1.0)
    sl.fit(data_img, y=cond)

    # Ensure model is fitted correctly
    assert sl.scores_ is not None
    assert sl.process_mask_ is not None

    # Perform transform on the same data
    transformed_scores = sl.transform(data_img)

    assert transformed_scores is not None
    assert transformed_scores.shape == (
        5,
        5,
        5,
    )
    assert transformed_scores.size > 0
