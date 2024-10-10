"""Test the _utils.param_validation module."""

import os
import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image, load
from sklearn.base import BaseEstimator

from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.param_validation import (
    MNI152_BRAIN_VOLUME,
    check_feature_screening,
    adjust_screening_percentile,
    check_threshold,
    get_mask_volume,
)

from nilearn.surface import vol_to_surf
from nilearn import datasets

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
)


def test_check_threshold():
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

    name = "threshold"
    # few not correctly formatted strings for 'threshold'
    wrong_thresholds = ["0.1", "10", "10.2.3%", "asdf%"]
    for wrong_threshold in wrong_thresholds:
        with pytest.raises(
            ValueError,
            match=f"{name}.+should be a number followed by the percent sign",
        ):
            check_threshold(wrong_threshold, matrix, fast_abs_percentile, name)

    threshold = object()
    with pytest.raises(
        TypeError, match=f"{name}.+should be either a number or a string"
    ):
        check_threshold(threshold, matrix, fast_abs_percentile, name)

    # Test threshold as int, threshold=2 should return as it is
    # since it is not string
    assert check_threshold(2, matrix, fast_abs_percentile) == 2

    # check whether raises a warning if given threshold is higher than expected
    with pytest.warns(UserWarning):
        check_threshold(3.0, matrix, fast_abs_percentile)

    # test with numpy scalar as argument
    threshold = 2.0
    threshold_numpy_scalar = np.float64(threshold)
    assert check_threshold(
        threshold, matrix, fast_abs_percentile
    ) == check_threshold(threshold_numpy_scalar, matrix, fast_abs_percentile)

    # Test for threshold provided as a percentile of the data
    # (str ending with a %)
    assert (
        1.0
        < check_threshold("50%", matrix, fast_abs_percentile, name=name)
        <= 2.0
    )


def test_get_mask_volume():
    # Test that hard-coded standard mask volume can be corrected computed
    if os.path.isfile(mni152_brain_mask):
        assert get_mask_volume(load(mni152_brain_mask)) == MNI152_BRAIN_VOLUME
    else:
        warnings.warn(f"Couldn't find {mni152_brain_mask} (for testing)")


def test_feature_screening(affine_eye):
    # dummy
    mask_img_data = np.zeros((182, 218, 182))
    mask_img_data[30:-30, 30:-30, 30:-30] = 1
    mask_img = Nifti1Image(mask_img_data, affine=affine_eye)

    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1, 10]:
            if screening_percentile == 100 or screening_percentile is None:
                assert (
                    check_feature_screening(
                        screening_percentile, mask_img, is_classif
                    )
                    is None
                )
            elif screening_percentile in {-1, 101}:
                with pytest.raises(ValueError):
                    check_feature_screening(
                        screening_percentile,
                        mask_img,
                        is_classif,
                    )
            elif screening_percentile == 20:
                assert isinstance(
                    check_feature_screening(
                        screening_percentile, mask_img, is_classif
                    ),
                    BaseEstimator,
                )


@pytest.mark.xfail
@pytest.mark.parametrize("screening_percentile", [1, 20, 90])
def test_screening_adjustment(affine_eye, screening_percentile):

    # get a mesh
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    # create sample mask image
    mask_img_data = np.zeros((182, 218, 182))
    mask_img_data[30:-30, 30:-30, 30:-30] = 1
    mask_img = Nifti1Image(mask_img_data, affine=affine_eye)
    mask_surf = vol_to_surf(
        img=mask_img,
        surf_mesh=fsaverage["pial_left"],
        inner_mesh=fsaverage["white_left"],
    )
    # create sample image
    img_data = np.random.randn(182, 218, 182)
    img = Nifti1Image(img_data, affine=affine_eye)
    img_surf = vol_to_surf(
        img=img,
        surf_mesh=fsaverage["pial_left"],
        inner_mesh=fsaverage["white_left"],
    )
    # get number of vertices in mask and mesh
    mask_n_vertices = get_mask_volume(mask_surf)
    mesh_n_vertices = fsaverage.mesh.n_vertices
    # get mask to mesh ratio
    mask_to_mesh_ratio = (mask_n_vertices / mesh_n_vertices) * 100
    # adjust screening percentile
    adjusted = adjust_screening_percentile(
        screening_percentile, mask_to_mesh_ratio, mesh_n_vertices
    )
    if screening_percentile == 100:
        assert adjusted == 100
    # if mask is smaller than given percentile, full mask is used
    if mask_to_mesh_ratio < screening_percentile:
        assert adjusted == 100
    # if mask is larger than given percentile, the percentile is adjusted to
    # the ratio of mesh to mask
    elif mask_to_mesh_ratio > screening_percentile:
        assert adjusted == screening_percentile * (
            mesh_n_vertices / mask_n_vertices
        )
    # if mask is equal to given percentile, the adjusted percentile is the
    # same as the given percentile
    elif mask_to_mesh_ratio == screening_percentile:
        assert adjusted == screening_percentile
