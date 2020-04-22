import os
import tempfile
import numpy as np
import nibabel as nb

from nilearn._utils.exceptions import DimensionError

import nibabel
import pytest

from nilearn.input_data.surf_labels_masker import (SurfLabelsMasker,
                                                   _all_files_check,
                                                   _get_single_surf,
                                                   surf_to_signals_labels)
from numpy.testing import assert_array_equal


def test_surf_labels_maskers():

    # 1D input case first
    labels_surf = np.array([0, 0, 1, 1])
    fake_1d_surf = np.ones(4)
    masker = SurfLabelsMasker(labels_surf=labels_surf)
    trans_surf = masker.fit_transform(fake_1d_surf)
    assert trans_surf.shape == (1, 1)
    assert np.sum(trans_surf) == 1

    # Make sure original array didn't get changed
    assert_array_equal(fake_1d_surf, np.ones(4))

    # Reverse transform case, should be able to recover
    # 1's in the second half, but where labels surf is 0
    # won't be able to recover that.
    back_to_original = masker.inverse_transform(trans_surf)
    expected = np.ones((4, 1))
    expected[:2] = 0
    assert_array_equal(back_to_original, expected)

    # 2D input case, as array
    fake_2d_surf = np.ones((4, 5))
    trans_surf = masker.fit_transform(fake_2d_surf)
    assert_array_equal(trans_surf, np.ones((5, 1)))

    # Inv transform
    back_to_original = masker.inverse_transform(trans_surf)
    expected = np.ones((4, 5))
    expected[:2] = 0
    assert_array_equal(back_to_original, expected)

    # Change labels surf
    labels_surf = np.array([1, 1, 2, 3])
    masker = SurfLabelsMasker(labels_surf=labels_surf)

    # Test load from list of 1D files
    files = [tempfile.mktemp(suffix='.nii') for _ in range(3)]
    for f in files:
        nii = nb.Nifti1Image(np.zeros((4, )), affine=None)
        nb.save(nii, f)
    trans_surf = masker.fit_transform(files)
    assert_array_equal(trans_surf, np.zeros((3, 3)))
    back_to_original = masker.inverse_transform(trans_surf)
    assert_array_equal(back_to_original, np.zeros((4, 3)))

    # Clean files
    for f in files:
        os.remove(f)

    # Test with passing a different background label
    masker = SurfLabelsMasker(labels_surf=labels_surf,
                              background_label=1)
    trans_surf = masker.fit_transform(fake_1d_surf)
    assert_array_equal(trans_surf, np.ones((1, 2)))
    back_to_original = masker.inverse_transform(trans_surf)
    expected = np.ones((4, 1))
    expected[:2] = 0
    assert_array_equal(back_to_original, expected)

    # Test with mask_labels_surf
    masker = SurfLabelsMasker(labels_surf=labels_surf,
                              background_label=1,
                              mask_labels_surf=np.array([0, 0, 0, 1]))
    trans_surf = masker.fit_transform(fake_1d_surf)
    assert_array_equal(trans_surf, np.ones((1, 1)))
    back_to_original = masker.inverse_transform(trans_surf)
    expected[2] = 0
    assert_array_equal(back_to_original, expected)

    # Check is fitted
    masker = SurfLabelsMasker(labels_surf=labels_surf)
    with pytest.raises(ValueError, match='has not been fitted. '):
        masker.inverse_transform(back_to_original)


def test_surf_labels_masker_reduction_strategies():

    test_values = [-2., -1., 0., 1., 2]
    surf_data = np.array(test_values * 2)
    labels = np.array([0] * 5 + [1] * 5, dtype=np.int8)

    # What SurfLabelsMasker should return for each reduction strategy
    expected_results = {"mean": np.mean(test_values),
                        "median": np.median(test_values),
                        "sum": np.sum(test_values),
                        "minimum": np.min(test_values),
                        "maximum": np.max(test_values),
                        "standard_deviation": np.std(test_values),
                        "variance": np.var(test_values)}

    for strategy, expected_result in expected_results.items():
        masker = SurfLabelsMasker(labels, strategy=strategy)
        result = masker.fit_transform(surf_data).squeeze()
        assert_array_equal(result, expected_result)

    with pytest.raises(ValueError, match="Invalid strategy 'TESTRAISE'"):
        SurfLabelsMasker(
            labels,
            strategy="TESTRAISE"
        )

    default_masker = SurfLabelsMasker(labels)
    assert default_masker.strategy == "mean"


def test_bad_input_surf_labels_masker_errors():

    # Bad label dimensions
    masker = SurfLabelsMasker(np.ones((10, 10)))
    pytest.raises(DimensionError, masker.fit)
    masker = SurfLabelsMasker(np.ones((5, 5, 6)))
    pytest.raises(DimensionError, masker.fit)

    # Bad mask dimensions
    mask = np.array([[1, 1], [0, 1]])
    masker = SurfLabelsMasker(np.ones(5), mask_labels_surf=mask)
    pytest.raises(DimensionError, masker.fit)

    # Mismatch labels and mask
    labels = np.array([1, 2, 3])
    mask = np.array([0, 0, 1, 1])
    masker = SurfLabelsMasker(labels, mask_labels_surf=mask)
    pytest.raises(ValueError, masker.fit)

    # Check background label bad input
    with pytest.raises(ValueError):
        SurfLabelsMasker(labels, background_label=6.6)
    with pytest.raises(ValueError):
        SurfLabelsMasker(labels, background_label='bad')

    # Check bad data to transform
    labels = np.array([1, 2, 3])
    masker = SurfLabelsMasker(labels)
    bad_surf_data = [np.ones(10), np.ones(6)]
    pytest.raises(ValueError, masker.fit_transform, bad_surf_data)

    bad_surf_data = np.ones((4, 4, 4))
    pytest.raises(DimensionError, masker.fit_transform, bad_surf_data)

    bad_surf_data = np.ones((5, 3))
    pytest.raises(ValueError, masker.fit_transform, bad_surf_data)

    # Check files case, with mimatch surf labels shape
    files = [tempfile.mktemp(suffix='.nii') for _ in range(3)]
    for f in range(len(files)):
        nii = nb.Nifti1Image(np.zeros((3 + f)), affine=None)
        nb.save(nii, files[f])

    pytest.raises(ValueError, masker.fit_transform, files)
    # Clean files
    for f in files:
        os.remove(f)


def test_surf_labels_masker_with_nans_and_infs():

    surf_data = np.array([np.nan, 0, np.inf, 1, 1, 1])
    labels = np.array([0, 0, 1, 1, 2, 2])

    masker = SurfLabelsMasker(labels)
    sig = masker.fit_transform(surf_data)
    print(sig)
    assert sig.shape == (1, 2)
    assert np.all(np.isfinite(sig))


def test_all_files_checks():

    files1 = [tempfile.mktemp(suffix='.nii') for _ in range(3)]
    files2 = ['as_strs' for _ in range(5)]

    assert(_all_files_check(files1))
    assert(_all_files_check(files2))

    not_files1 = np.array((10, 6))
    not_files2 = [np.ones(5), np.ones(5)]

    assert (_all_files_check(not_files1) is False)
    assert (_all_files_check(not_files2) is False)


def test_get_single_surf():

    # Not files case
    surfs_2d = np.ones((5, 2))
    for n in range(2):
        surf = _get_single_surf(surfs_2d, n, False)
        assert_array_equal(surf, np.ones(5))

    # Files case - pretend arrays are files
    surf_files = np.ones((2, 5))
    for n in range(2):
        surf = _get_single_surf(surf_files, n, True)
        assert_array_equal(surf, np.ones(5))


def test_surf_to_signals_labels_bad_strategy():

    with pytest.raises(ValueError):

        surf_to_signals_labels(np.ones((5, 5)), np.ones((5)),
                               strategy='magic')















    

    


    

