""" Test Region Extractor and its functions """

import numpy as np
import nibabel

from nose.tools import assert_raises

from nilearn.regions.region_extractor import (estimate_apply_threshold_to_maps,
                                              break_connected_components,
                                              RegionExtractor)
from nilearn.image import iter_img

from nilearn._utils.testing import assert_raises_regex, generate_maps


def _make_random_data(shape):
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    data_rng = rng.normal(size=shape)
    img = nibabel.Nifti1Image(data_rng, affine)
    data = img.get_data()
    return img, data


def test_validity_threshold_types_in_estimate_apply_threshold_to_maps():

    shape = (6, 8, 10)
    maps = generate_maps(shape, n_regions=2)
    map_0 = maps[0]

    # Test whether function estimate_apply_threshold_to_maps
    # raises the same error using t-value higher than the actual
    t_value = 2.
    assert_raises_regex(ValueError,
                        "The value given to threshold "
                        "statistical maps must not exceed 1. "
                        "You provided threshold=%s " % t_value,
                        estimate_apply_threshold_to_maps,
                        map_0, threshold=t_value,
                        thresholding_strategy=None)

    invalid_threshold_estimates = ['percent', 'ratio', 'auto']
    for invalid_estimate in invalid_threshold_estimates:
        assert_raises(ValueError, estimate_apply_threshold_to_maps,
                      map_0, threshold='auto',
                      thresholding_strategy=invalid_estimate)

    invalid_threshold_value = ['10', 'some_value']
    for thr in invalid_threshold_value:
        assert_raises(ValueError, estimate_apply_threshold_to_maps,
                      map_0, threshold=thr)


def test_passing_estimate_apply_threshold_to_maps():
    # smoke test for function estimate_apply_threshold_to_maps to check
    # whether passes through valid threshold inputs
    shape = (10, 20, 30)
    maps = generate_maps(shape, n_regions=4)
    map_0 = maps[0]
    valid_threshold_estimates = ['percentile', 'ratio_n_voxels']
    for map_, strategy in zip(iter_img(map_0), valid_threshold_estimates):
        thr_maps = estimate_apply_threshold_to_maps(map_, threshold='auto',
                                                    thresholding_strategy=strategy)

    # smoke test to check whether function estimate_apply_threshold_to_maps
    # passes through input t-value
    t_value_thresholds = [0.5, 0.8]
    for t_val in t_value_thresholds:
        thr_maps = estimate_apply_threshold_to_maps(map_0, threshold=t_val,
                                                    thresholding_strategy=None)


def test_validity_extract_types_in_break_connected_components():
    shape = (91, 109, 91)
    n_regions = 2
    maps = generate_maps(shape, n_regions)
    map_0 = maps[0]

    # test whether same error raises as expected when given invalid inputs to
    # extract
    valid_extract_names = ['connected_components', 'local_regions']
    message = ("'extract_type' should be given "
               "either of these \\['connected_components', 'local_regions'\\]")
    for img in iter_img(map_0):
        assert_raises_regex(ValueError,
                            message,
                            break_connected_components,
                            img, min_size=40,
                            extract_type='connect_regions')

    # test whether same error raises as expected when you submit 4D image
    assert_raises_regex(ValueError,
                        "A 3D Nifti image or path to a 3D image should be submitted.",
                        break_connected_components,
                        map_0, extract_type='local_regions')

    # test whether error raises when there is an invalid inputs to extract
    invalid_extract_type = ['local', 'asdf', 10]
    for type_ in invalid_extract_type:
        assert_raises(ValueError, break_connected_components,
                      img, extract_type=type_)


def test_passing_break_connected_components():
    shape = (91, 109, 91)
    n_regions = 4
    maps = generate_maps(shape, n_regions)
    map_0 = maps[0]

    valid_extract_types = ['connected_components', 'local_regions']
    # smoke test for function extract_regions
    for map_, type_ in zip(iter_img(map_0), valid_extract_types):
        regions_extracted = break_connected_components(map_, min_size=30,
                                                       extract_type=type_,
                                                       peak_local_smooth=3)


def test_passing_RegionExtractor_object():
    imgs = []
    shape = (91, 109, 91)
    n_regions = 2
    n_subjects = 5
    maps = generate_maps(shape, n_regions)
    map_0 = maps[0]

    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(np.ones((shape), dtype=np.int8), affine)

    # smoke test fit() and fit_transform() with giving inputs
    region_extractor = RegionExtractor(map_0, mask=mask_img)
    # smoke test fit() function
    region_extractor.fit()

    shape = (91, 109, 91, 7)
    for id_ in range(n_subjects):
        img, data = _make_random_data(shape)
        imgs.append(img)
    # smoke test fit_transform() function
    region_extractor.fit_transform(imgs)
