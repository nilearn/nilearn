""" Test Region Extractor and its functions """

import numpy as np
import nibabel

from nose.tools import assert_equal, assert_true, assert_not_equal

from nilearn.regions import connected_regions, RegionExtractor
from nilearn.regions.region_extractor import _threshold_maps_ratio

from nilearn._utils.testing import assert_raises_regex, generate_maps


def _make_random_data(shape):
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    data_rng = rng.normal(size=shape)
    img = nibabel.Nifti1Image(data_rng, affine)
    data = img.get_data()
    return img, data


def test_invalid_thresholds_in_threshold_maps_ratio():
    maps, _ = generate_maps((10, 11, 12), n_regions=2)

    for invalid_threshold in ['80%', 'auto', -1.0]:
        assert_raises_regex(ValueError,
                            "threshold given as ratio to the number of voxels must "
                            "be Real number and should be positive and between 0 and "
                            "total number of maps i.e. n_maps={0}. "
                            "You provided {1}".format(maps.shape[-1], invalid_threshold),
                            _threshold_maps_ratio,
                            maps, threshold=invalid_threshold)


def test_nans_threshold_maps_ratio():
    maps, _ = generate_maps((10, 10, 10), n_regions=2)
    data = maps.get_data()
    data[:, :, 0] = np.nan

    maps_img = nibabel.Nifti1Image(data, np.eye(4))
    thr_maps = _threshold_maps_ratio(maps_img, threshold=0.8)


def test_threshold_maps_ratio():
    # smoke test for function _threshold_maps_ratio with randomly
    # generated maps

    # make sure that n_regions (4th dimension) are kept same even
    # in thresholded image
    maps, _ = generate_maps((6, 8, 10), n_regions=3)
    thr_maps = _threshold_maps_ratio(maps, threshold=1.0)
    assert_true(thr_maps.shape[-1] == maps.shape[-1])

    # check that the size should be same for 3D image
    # before and after thresholding
    img = np.zeros((30, 30, 30)) + 0.1 * np.random.randn(30, 30, 30)
    img = nibabel.Nifti1Image(img, affine=np.eye(4))
    thr_maps_3d = _threshold_maps_ratio(img, threshold=0.5)
    assert_true(img.shape == thr_maps_3d.shape)


def test_invalids_extract_types_in_connected_regions():
    maps, _ = generate_maps((10, 11, 12), n_regions=2)
    valid_names = ['connected_components', 'local_regions']

    # test whether same error raises as expected when invalid inputs
    # are given to extract_type in connected_regions function
    message = ("'extract_type' should be {0}")
    for invalid_extract_type in ['connect_region', 'local_regios']:
        assert_raises_regex(ValueError,
                            message.format(valid_names),
                            connected_regions,
                            maps, extract_type=invalid_extract_type)


def test_connected_regions():
    # 4D maps
    n_regions = 4
    maps, _ = generate_maps((30, 30, 30), n_regions=n_regions)
    # 3D maps
    map_img = np.zeros((30, 30, 30)) + 0.1 * np.random.randn(30, 30, 30)
    map_img = nibabel.Nifti1Image(map_img, affine=np.eye(4))

    # smoke test for function connected_regions and also to check
    # if the regions extracted should be equal or more than already present.
    # 4D image case
    for extract_type in ['connected_components', 'local_regions']:
        connected_extraction_img, index = connected_regions(maps, min_region_size=10,
                                                            extract_type=extract_type)
        assert_true(connected_extraction_img.shape[-1] >= n_regions)
        assert_true(index, np.ndarray)
        # For 3D images regions extracted should be more than equal to one
        connected_extraction_3d_img, _ = connected_regions(map_img, min_region_size=10,
                                                           extract_type=extract_type)
        assert_true(connected_extraction_3d_img.shape[-1] >= 1)


def test_invalid_threshold_strategies():
    maps, _ = generate_maps((6, 8, 10), n_regions=1)

    extract_strategy_check = RegionExtractor(maps, thresholding_strategy='n_')
    valid_strategies = ['ratio_n_voxels', 'img_value', 'percentile']
    assert_raises_regex(ValueError,
                        "'thresholding_strategy' should be either of "
                        "these".format(valid_strategies),
                        extract_strategy_check.fit)


def test_threshold_as_none_and_string_cases():
    maps, _ = generate_maps((6, 8, 10), n_regions=1)

    extract_thr_none_check = RegionExtractor(maps, threshold=None)
    assert_raises_regex(ValueError,
                        "The given input to threshold is not valid.",
                        extract_thr_none_check.fit)
    extract_thr_string_check = RegionExtractor(maps, threshold='30%')
    assert_raises_regex(ValueError,
                        "The given input to threshold is not valid.",
                        extract_thr_string_check.fit)


def test_region_extractor_fit_and_transform():
    n_regions = 9
    n_subjects = 5
    maps, mask_img = generate_maps((40, 40, 40), n_regions=n_regions)

    # smoke test to RegionExtractor with thresholding_strategy='ratio_n_voxels'
    extract_ratio = RegionExtractor(maps, threshold=0.2,
                                    thresholding_strategy='ratio_n_voxels')
    extract_ratio.fit()
    assert_not_equal(extract_ratio.regions_img_, '')
    assert_true(extract_ratio.regions_img_.shape[-1] >= 9)

    # smoke test with threshold=string and strategy=percentile
    extractor = RegionExtractor(maps, threshold=30,
                                thresholding_strategy='percentile',
                                mask_img=mask_img)
    extractor.fit()
    assert_true(extractor.index_, np.ndarray)
    assert_not_equal(extractor.regions_img_, '')
    assert_true(extractor.regions_img_.shape[-1] >= 9)

    n_regions_extracted = extractor.regions_img_.shape[-1]
    shape = (91, 109, 91, 7)
    expected_signal_shape = (7, n_regions_extracted)
    for id_ in range(n_subjects):
        img, data = _make_random_data(shape)
        # smoke test NiftiMapsMasker transform inherited in Region Extractor
        signal = extractor.transform(img)
        assert_equal(expected_signal_shape, signal.shape)
