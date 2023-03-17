"""Test Region Extractor and its functions"""

import nibabel
import numpy as np
import pytest
from nilearn._utils.data_gen import generate_labeled_regions, generate_maps
from nilearn._utils.exceptions import DimensionError
from nilearn.image import get_data
from nilearn.regions import (
    RegionExtractor,
    connected_label_regions,
    connected_regions,
)
from nilearn.regions.region_extractor import (
    _remove_small_regions,
    _threshold_maps_ratio,
)
from scipy.ndimage import label


def _make_random_data(shape):
    affine = np.eye(4)
    rng = np.random.RandomState(42)
    data_rng = rng.normal(size=shape)
    img = nibabel.Nifti1Image(data_rng, affine)
    data = get_data(img)
    return img, data


def test_invalid_thresholds_in_threshold_maps_ratio():
    maps, _ = generate_maps((10, 11, 12), n_regions=2)

    for invalid_threshold in ["80%", "auto", -1.0]:
        with pytest.raises(
            ValueError,
            match="threshold given as ratio to the number of voxels must "
            "be Real number and should be positive "
            "and between 0 and total number of maps "
            f"i.e. n_maps={maps.shape[-1]}. "
            f"You provided {invalid_threshold}",
        ):
            _threshold_maps_ratio(maps, threshold=invalid_threshold)


def test_nans_threshold_maps_ratio():
    maps, _ = generate_maps((10, 10, 10), n_regions=2)
    data = get_data(maps)
    data[:, :, 0] = np.nan

    maps_img = nibabel.Nifti1Image(data, np.eye(4))
    _threshold_maps_ratio(maps_img, threshold=0.8)


def test_threshold_maps_ratio():
    # smoke test for function _threshold_maps_ratio with randomly
    # generated maps

    rng = np.random.RandomState(42)

    maps, _ = generate_maps((6, 8, 10), n_regions=3)

    # test that there is no side effect
    get_data(maps)[:3] = 100
    maps_data = get_data(maps).copy()
    thr_maps = _threshold_maps_ratio(maps, threshold=1.0)
    np.testing.assert_array_equal(get_data(maps), maps_data)

    # make sure that n_regions (4th dimension) are kept same even
    # in thresholded image
    assert thr_maps.shape[-1] == maps.shape[-1]

    # check that the size should be same for 3D image
    # before and after thresholding
    img = np.zeros((30, 30, 30)) + 0.1 * rng.standard_normal(size=(30, 30, 30))
    img = nibabel.Nifti1Image(img, affine=np.eye(4))
    thr_maps_3d = _threshold_maps_ratio(img, threshold=0.5)
    assert img.shape == thr_maps_3d.shape


def test_invalids_extract_types_in_connected_regions():
    maps, _ = generate_maps((10, 11, 12), n_regions=2)
    valid_names = ["connected_components", "local_regions"]

    # test whether same error raises as expected when invalid inputs
    # are given to extract_type in connected_regions function
    message = f"'extract_type' should be {valid_names}"
    for invalid_extract_type in ["spam", "eggs"]:
        with pytest.raises(ValueError, match=message):
            connected_regions(maps, extract_type=invalid_extract_type)


def test_connected_regions():
    rng = np.random.RandomState(42)

    # 4D maps
    n_regions = 4
    maps, mask_img = generate_maps((30, 30, 30), n_regions=n_regions)
    # 3D maps
    map_img = np.zeros((30, 30, 30)) + 0.1 * rng.standard_normal(
        size=(30, 30, 30)
    )
    map_img = nibabel.Nifti1Image(map_img, affine=np.eye(4))

    # smoke test for function connected_regions and also to check
    # if the regions extracted should be equal or more than already present.
    # 4D image case
    for extract_type in ["connected_components", "local_regions"]:
        connected_extraction_img, index = connected_regions(
            maps, min_region_size=10, extract_type=extract_type
        )
        assert connected_extraction_img.shape[-1] >= n_regions
        assert index, np.ndarray
        # For 3D images regions extracted should be more than equal to one
        connected_extraction_3d_img, _ = connected_regions(
            map_img, min_region_size=10, extract_type=extract_type
        )
        assert connected_extraction_3d_img.shape[-1] >= 1

    # Test input mask_img
    mask = get_data(mask_img)
    mask[1, 1, 1] = 0
    extraction_with_mask_img, index = connected_regions(
        maps, mask_img=mask_img
    )
    assert extraction_with_mask_img.shape[-1] >= 1

    extraction_without_mask_img, index = connected_regions(maps)
    assert np.all(get_data(extraction_with_mask_img)[mask == 0] == 0.0)
    assert not np.all(get_data(extraction_without_mask_img)[mask == 0] == 0.0)

    # mask_img with different shape
    mask = np.zeros(shape=(10, 11, 12), dtype="uint8")
    mask[1:-1, 1:-1, 1:-1] = 1
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]
    )
    mask_img = nibabel.Nifti1Image(mask, affine=affine)
    extraction_not_same_fov_mask, _ = connected_regions(
        maps, mask_img=mask_img
    )
    assert maps.shape[:3] == extraction_not_same_fov_mask.shape[:3]
    assert mask_img.shape != extraction_not_same_fov_mask.shape[:3]

    extraction_not_same_fov, _ = connected_regions(maps)
    assert np.sum(get_data(extraction_not_same_fov) == 0) > np.sum(
        get_data(extraction_not_same_fov_mask) == 0
    )


def test_invalid_threshold_strategies():
    maps, _ = generate_maps((6, 8, 10), n_regions=1)

    extract_strategy_check = RegionExtractor(maps, thresholding_strategy="n_")
    with pytest.raises(
        ValueError,
        match="'thresholding_strategy' should be ",
    ):
        extract_strategy_check.fit()


def test_threshold_as_none_and_string_cases():
    maps, _ = generate_maps((6, 8, 10), n_regions=1)

    extract_thr_none_check = RegionExtractor(maps, threshold=None)
    with pytest.raises(
        ValueError, match="The given input to threshold is not valid."
    ):
        extract_thr_none_check.fit()
    extract_thr_string_check = RegionExtractor(maps, threshold="30%")
    with pytest.raises(
        ValueError, match="The given input to threshold is not valid."
    ):
        extract_thr_string_check.fit()


def test_region_extractor_fit_and_transform():
    n_regions = 9
    n_subjects = 5
    maps, mask_img = generate_maps((40, 40, 40), n_regions=n_regions)

    # Test maps are zero in the mask
    mask_data = get_data(mask_img)
    mask_data[1, 1, 1] = 0
    extractor_without_mask = RegionExtractor(maps)
    extractor_without_mask.fit()
    extractor_with_mask = RegionExtractor(maps, mask_img=mask_img)
    extractor_with_mask.fit()
    assert not np.all(
        get_data(extractor_without_mask.regions_img_)[mask_data == 0] == 0.0
    )
    assert np.all(
        get_data(extractor_with_mask.regions_img_)[mask_data == 0] == 0.0
    )

    # smoke test to RegionExtractor with thresholding_strategy='ratio_n_voxels'
    extract_ratio = RegionExtractor(
        maps, threshold=0.2, thresholding_strategy="ratio_n_voxels"
    )
    extract_ratio.fit()
    assert extract_ratio.regions_img_ != ""
    assert extract_ratio.regions_img_.shape[-1] >= 9

    # smoke test with threshold=string and strategy=percentile
    extractor = RegionExtractor(
        maps,
        threshold=30,
        thresholding_strategy="percentile",
        mask_img=mask_img,
    )
    extractor.fit()
    assert extractor.index_, np.ndarray
    assert extractor.regions_img_ != ""
    assert extractor.regions_img_.shape[-1] >= 9

    n_regions_extracted = extractor.regions_img_.shape[-1]
    shape = (91, 109, 91, 7)
    expected_signal_shape = (7, n_regions_extracted)
    for _ in range(n_subjects):
        img, _ = _make_random_data(shape)
        # smoke test NiftiMapsMasker transform inherited in Region Extractor
        signal = extractor.transform(img)
        assert expected_signal_shape == signal.shape

    # smoke test with high resolution image
    maps, mask_img = generate_maps(
        (20, 20, 20), n_regions=n_regions, affine=0.2 * np.eye(4)
    )

    extract_ratio = RegionExtractor(
        maps,
        thresholding_strategy="ratio_n_voxels",
        smoothing_fwhm=0.6,
        min_region_size=0.4,
    )
    extract_ratio.fit()
    assert extract_ratio.regions_img_ != ""
    assert extract_ratio.regions_img_.shape[-1] >= 9

    # smoke test with zeros on the diagonal of the affine
    affine = np.eye(4)
    affine[[0, 1]] = affine[[1, 0]]  # permutes first and second lines
    maps, mask_img = generate_maps(
        (40, 40, 40), n_regions=n_regions, affine=affine
    )

    extract_ratio = RegionExtractor(
        maps, threshold=0.2, thresholding_strategy="ratio_n_voxels"
    )
    extract_ratio.fit()
    assert extract_ratio.regions_img_ != ""
    assert extract_ratio.regions_img_.shape[-1] >= 9


def test_error_messages_connected_label_regions():
    shape = (13, 11, 12)
    affine = np.eye(4)
    n_regions = 2
    labels_img = generate_labeled_regions(
        shape, affine=affine, n_regions=n_regions
    )
    with pytest.raises(
        ValueError, match="Expected 'min_size' to be specified as integer."
    ):
        connected_label_regions(labels_img=labels_img, min_size="a")
    with pytest.raises(
        ValueError, match="'connect_diag' must be specified as True or False."
    ):
        connected_label_regions(labels_img=labels_img, connect_diag=None)


def test_remove_small_regions():
    data = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        ]
    )
    # To remove small regions, data should be labelled
    label_map, n_labels = label(data)
    sum_label_data = np.sum(label_map)

    affine = np.eye(4)
    min_size = 10
    # data can be act as mask_data to identify regions in label_map because
    # features in label_map are built upon non-zeros in data
    index = np.arange(n_labels + 1)
    removed_data = _remove_small_regions(label_map, index, affine, min_size)
    sum_removed_data = np.sum(removed_data)

    assert sum_removed_data < sum_label_data


def test_connected_label_regions():
    shape = (13, 11, 12)
    affine = np.eye(4)
    n_regions = 9
    labels_img = generate_labeled_regions(
        shape, affine=affine, n_regions=n_regions
    )
    labels_data = get_data(labels_img)
    n_labels_wo_reg_ext = len(np.unique(labels_data))

    # region extraction without specifying min_size
    extracted_regions_on_labels_img = connected_label_regions(labels_img)
    extracted_regions_labels_data = get_data(extracted_regions_on_labels_img)
    n_labels_wo_min = len(np.unique(extracted_regions_labels_data))

    assert n_labels_wo_reg_ext < n_labels_wo_min

    # with specifying min_size
    extracted_regions_with_min = connected_label_regions(
        labels_img, min_size=100
    )
    extracted_regions_with_min_data = get_data(extracted_regions_with_min)
    n_labels_with_min = len(np.unique(extracted_regions_with_min_data))

    assert n_labels_wo_min > n_labels_with_min

    # Test connect_diag=False
    ext_reg_without_connect_diag = connected_label_regions(
        labels_img, connect_diag=False
    )
    data_wo_connect_diag = get_data(ext_reg_without_connect_diag)
    n_labels_wo_connect_diag = len(np.unique(data_wo_connect_diag))
    assert n_labels_wo_connect_diag > n_labels_wo_reg_ext

    # If min_size is large and if all the regions are removed then empty image
    # will be returned
    extract_reg_min_size_large = connected_label_regions(
        labels_img, min_size=500
    )
    assert np.unique(get_data(extract_reg_min_size_large)) == 0

    # Test the names of the brain regions given in labels.
    # Test labels for 9 regions in n_regions
    labels = [f"region_{x}" for x in "abcdefghi"]

    # If labels are provided, first return will contain extracted labels image
    # and second return will contain list of new names generated based on same
    # name with assigned on both hemispheres for example.
    _, new_labels = connected_label_regions(
        labels_img, min_size=100, labels=labels
    )
    # The length of new_labels returned can differ depending upon min_size. If
    # min_size given is more small regions can be removed therefore newly
    # generated labels can be less than original size of labels. Or if min_size
    # is less then newly generated labels can be more.

    # We test here whether labels returned are empty or not.
    assert new_labels != ""
    assert len(new_labels) <= len(labels)

    # labels given in numpy array
    labels = np.asarray(labels)
    _, new_labels2 = connected_label_regions(labels_img, labels=labels)
    assert new_labels != ""
    # By default min_size is less, so newly generated labels can be more.
    assert len(new_labels2) >= len(labels)

    # If number of labels provided are wrong (which means less than number of
    # unique labels in labels_img), then we raise an error

    # Test whether error raises
    unique_labels = set(np.unique(np.asarray(get_data(labels_img))))
    unique_labels.remove(0)

    # labels given are less than n_regions=9
    provided_labels = [f"region_{x}" for x in "acfghi"]

    assert len(provided_labels) < len(unique_labels)

    with pytest.raises(ValueError):
        connected_label_regions(labels_img, labels=provided_labels)

    # Test if unknown/negative integers are provided as labels in labels_img,
    # we raise an error and test the same whether error is raised.
    # Introduce data type of float
    # see issue: https://github.com/nilearn/nilearn/issues/2580
    labels_data = np.zeros(shape, dtype=np.float32)
    h0 = shape[0] // 2
    h1 = shape[1] // 2
    h2 = shape[2] // 2
    labels_data[:h0, :h1, :h2] = 1
    labels_data[:h0, :h1, h2:] = 2
    labels_data[:h0, h1:, :h2] = 3
    labels_data[:h0, h1:, h2:] = -4
    labels_data[h0:, :h1, :h2] = 5
    labels_data[h0:, :h1, h2:] = 6
    labels_data[h0:, h1:, :h2] = np.nan
    labels_data[h0:, h1:, h2:] = np.inf

    neg_labels_img = nibabel.Nifti1Image(labels_data, affine)
    with pytest.raises(ValueError):
        connected_label_regions(labels_img=neg_labels_img)

    # If labels_img provided is 4D Nifti image, then test whether error is
    # raised or not. Since this function accepts only 3D image.
    labels_4d_data = np.zeros((shape) + (2,))
    labels_data[h0:, h1:, :h2] = 0
    labels_data[h0:, h1:, h2:] = 0
    labels_4d_data[..., 0] = labels_data
    labels_4d_data[..., 1] = labels_data
    labels_img_4d = nibabel.Nifti1Image(labels_4d_data, np.eye(4))
    with pytest.raises(DimensionError):
        connected_label_regions(labels_img=labels_img_4d)

    # Test if labels (or names to regions) given is a string without a list.
    # Then, we expect it to be split to regions extracted and returned as list.
    labels_in_str = "region_a"
    labels_img_in_str = generate_labeled_regions(
        shape, affine=affine, n_regions=1
    )
    _, new_labels = connected_label_regions(
        labels_img_in_str, labels=labels_in_str
    )
    assert isinstance(new_labels, list)

    # If user has provided combination of labels, then function passes without
    # breaking and new labels are returned based upon given labels and should
    # be equal or more based on regions extracted
    combined_labels = [
        "region_a",
        "1",
        "region_b",
        "2",
        "region_c",
        "3",
        "region_d",
        "4",
        "region_e",
    ]
    _, new_labels = connected_label_regions(labels_img, labels=combined_labels)
    assert len(new_labels) >= len(combined_labels)
