"""Test Region Extractor and its functions."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from scipy.ndimage import label

from nilearn._utils.data_gen import generate_labeled_regions, generate_maps
from nilearn._utils.exceptions import DimensionError
from nilearn.conftest import _affine_eye, _img_4d_zeros, _shape_3d_default
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

N_REGIONS = 3

MAP_SHAPE = (30, 30, 30)


@pytest.fixture
def negative_regions():
    return False


@pytest.fixture(scope="module")
def dummy_map():
    """Generate a small dummy map.

    Use for error testing
    """
    return generate_maps(shape=(6, 6, 6), n_regions=N_REGIONS)[0]


@pytest.fixture
def labels_img():
    n_regions = 9  # DO NOT CHANGE (some tests expect this value)
    return generate_labeled_regions(
        shape=_shape_3d_default(), affine=_affine_eye(), n_regions=n_regions
    )


@pytest.fixture
def maps(negative_regions):
    return generate_maps(
        shape=MAP_SHAPE,
        n_regions=N_REGIONS,
        random_state=42,
        negative_regions=negative_regions,
    )[0]


@pytest.fixture
def maps_and_mask():
    return generate_maps(shape=MAP_SHAPE, n_regions=N_REGIONS, random_state=42)


@pytest.fixture
def map_img_3d(rng):
    map_img = np.zeros(MAP_SHAPE) + 0.1 * rng.standard_normal(size=MAP_SHAPE)
    return Nifti1Image(map_img, affine=_affine_eye())


@pytest.mark.parametrize("invalid_threshold", ["80%", "auto", -1.0])
def test_invalid_thresholds_in_threshold_maps_ratio(
    dummy_map, invalid_threshold
):
    with pytest.raises(
        ValueError,
        match="threshold given as ratio to the number of voxels must "
        "be Real number and should be positive "
        "and between 0 and total number of maps "
        f"i.e. n_maps={dummy_map.shape[-1]}. "
        f"You provided {invalid_threshold}",
    ):
        _threshold_maps_ratio(maps_img=dummy_map, threshold=invalid_threshold)


def test_nans_threshold_maps_ratio(maps, affine_eye):
    data = get_data(maps)
    data[:, :, 0] = np.nan

    maps_img = Nifti1Image(data, affine_eye)
    _threshold_maps_ratio(maps_img, threshold=0.8)


def test_threshold_maps_ratio(maps):
    """Check _threshold_maps_ratio with randomly generated maps."""
    # test that there is no side effect
    get_data(maps)[:3] = 100
    maps_data = get_data(maps).copy()
    thr_maps = _threshold_maps_ratio(maps, threshold=1.0)
    np.testing.assert_array_equal(get_data(maps), maps_data)

    # make sure that n_regions (4th dimension) are kept same even
    # in thresholded image
    assert thr_maps.shape[-1] == maps.shape[-1]


def test_threshold_maps_ratio_3d(map_img_3d):
    """Check size is the same for 3D image before and after thresholding."""
    thr_maps_3d = _threshold_maps_ratio(map_img_3d, threshold=0.5)
    assert map_img_3d.shape == thr_maps_3d.shape


@pytest.mark.parametrize("invalid_extract_type", ["spam", 1])
def test_invalids_extract_types_in_connected_regions(
    dummy_map, invalid_extract_type
):
    valid_names = ["connected_components", "local_regions"]
    message = f"'extract_type' should be {valid_names}"
    with pytest.raises(ValueError, match=message):
        connected_regions(dummy_map, extract_type=invalid_extract_type)


@pytest.mark.parametrize(
    "extract_type", ["connected_components", "local_regions"]
)
def test_connected_regions_4d(maps, extract_type):
    """Regions extracted should be equal or more than already present."""
    connected_extraction_img, index = connected_regions(
        maps, min_region_size=10, extract_type=extract_type
    )
    assert connected_extraction_img.shape[-1] >= N_REGIONS
    assert index, np.ndarray


@pytest.mark.parametrize(
    "extract_type", ["connected_components", "local_regions"]
)
def test_connected_regions_3d(map_img_3d, extract_type):
    """For 3D images regions extracted should be more than equal to 1."""
    connected_extraction_3d_img, _ = connected_regions(
        maps_img=map_img_3d, min_region_size=10, extract_type=extract_type
    )
    assert connected_extraction_3d_img.shape[-1] >= 1


def test_connected_regions_different_results_with_different_mask_images(
    maps_and_mask,
):
    maps, mask_img = maps_and_mask
    # Test input mask_img
    mask = get_data(mask_img)
    mask[1, 1, 1] = 0

    extraction_with_mask_img, _ = connected_regions(maps, mask_img=mask_img)

    assert extraction_with_mask_img.shape[-1] >= 1

    extraction_without_mask_img, _ = connected_regions(maps)

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
    mask_img = Nifti1Image(mask, affine=affine)
    extraction_not_same_fov_mask, _ = connected_regions(
        maps, mask_img=mask_img
    )

    assert maps.shape[:3] == extraction_not_same_fov_mask.shape[:3]
    assert mask_img.shape != extraction_not_same_fov_mask.shape[:3]

    extraction_not_same_fov, _ = connected_regions(maps)

    assert np.sum(get_data(extraction_not_same_fov) == 0) > np.sum(
        get_data(extraction_not_same_fov_mask) == 0
    )


def test_invalid_threshold_strategies(dummy_map):
    extract_strategy_check = RegionExtractor(
        dummy_map, thresholding_strategy="n_"
    )

    with pytest.raises(
        ValueError,
        match="'thresholding_strategy' should be ",
    ):
        extract_strategy_check.fit()


@pytest.mark.parametrize("threshold", [None, "30%"])
def test_threshold_as_none_and_string_cases(dummy_map, threshold):
    to_check = RegionExtractor(dummy_map, threshold=threshold)

    with pytest.raises(
        ValueError, match="The given input to threshold is not valid."
    ):
        to_check.fit()


def test_region_extractor_fit_and_transform(maps_and_mask):
    maps, mask_img = maps_and_mask

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


def test_region_extractor_strategy_ratio_n_voxels(maps):
    extract_ratio = RegionExtractor(
        maps, threshold=0.2, thresholding_strategy="ratio_n_voxels"
    )
    extract_ratio.fit()

    assert extract_ratio.regions_img_ != ""
    assert extract_ratio.regions_img_.shape[-1] >= N_REGIONS


@pytest.mark.parametrize("negative_regions", [True])
def test_region_extractor_two_sided(maps):
    threshold = 0.4
    thresholding_strategy = "img_value"
    min_region_size = 5

    extract_ratio1 = RegionExtractor(
        maps,
        threshold=threshold,
        thresholding_strategy=thresholding_strategy,
        two_sided=False,
        min_region_size=min_region_size,
        extractor="connected_components",
    )
    extract_ratio1.fit()

    extract_ratio2 = RegionExtractor(
        maps,
        threshold=threshold,
        thresholding_strategy=thresholding_strategy,
        two_sided=True,
        min_region_size=min_region_size,
        extractor="connected_components",
    )

    extract_ratio2.fit()

    assert not np.array_equal(
        np.unique(extract_ratio1.regions_img_.get_fdata()),
        np.unique(extract_ratio2.regions_img_.get_fdata()),
    )


def test_region_extractor_strategy_percentile(maps_and_mask):
    maps, mask_img = maps_and_mask

    extractor = RegionExtractor(
        maps,
        threshold=30,
        thresholding_strategy="percentile",
        mask_img=mask_img,
    )
    extractor.fit()

    assert extractor.index_, np.ndarray
    assert extractor.regions_img_ != ""
    assert extractor.regions_img_.shape[-1] >= N_REGIONS

    n_regions_extracted = extractor.regions_img_.shape[-1]
    shape = (91, 109, 91, 7)
    expected_signal_shape = (7, n_regions_extracted)
    n_subjects = 3
    for _ in range(n_subjects):
        # smoke test NiftiMapsMasker transform inherited in Region Extractor
        signal = extractor.transform(_img_4d_zeros(shape=shape))

        assert expected_signal_shape == signal.shape


def test_region_extractor_high_resolution_image(affine_eye):
    n_regions = 9
    maps, _ = generate_maps(
        shape=MAP_SHAPE, n_regions=n_regions, affine=0.2 * affine_eye
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


def test_region_extractor_zeros_affine_diagonal(affine_eye):
    n_regions = 9
    affine = affine_eye
    affine[[0, 1]] = affine[[1, 0]]  # permutes first and second lines
    maps, _ = generate_maps(
        shape=[40, 40, 40], n_regions=n_regions, affine=affine, random_state=42
    )

    extract_ratio = RegionExtractor(
        maps, threshold=0.2, thresholding_strategy="ratio_n_voxels"
    )
    extract_ratio.fit()

    assert extract_ratio.regions_img_ != ""
    assert extract_ratio.regions_img_.shape[-1] >= n_regions


def test_error_messages_connected_label_regions(labels_img):
    with pytest.raises(
        ValueError, match="Expected 'min_size' to be specified as integer."
    ):
        connected_label_regions(labels_img=labels_img, min_size="a")
    with pytest.raises(
        ValueError, match="'connect_diag' must be specified as True or False."
    ):
        connected_label_regions(labels_img=labels_img, connect_diag=None)


def test_remove_small_regions(affine_eye):
    data = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        ]
    )
    # To remove small regions, data should be labeled
    label_map, n_labels = label(data)
    sum_label_data = np.sum(label_map)

    min_size = 10
    # data can be act as mask_data to identify regions in label_map because
    # features in label_map are built upon non-zeros in data
    removed_data = _remove_small_regions(label_map, affine_eye, min_size)
    sum_removed_data = np.sum(removed_data)

    assert sum_removed_data < sum_label_data


def test_connected_label_regions(labels_img):
    labels_data = get_data(labels_img)
    n_labels_without_region_extraction = len(np.unique(labels_data))

    # extract region without specifying min_size
    extracted_regions_on_labels_img = connected_label_regions(labels_img)
    extracted_regions_labels_data = get_data(extracted_regions_on_labels_img)
    n_labels_without_min = len(np.unique(extracted_regions_labels_data))

    assert n_labels_without_region_extraction < n_labels_without_min

    # with specifying min_size
    extracted_regions_with_min = connected_label_regions(
        labels_img, min_size=100
    )
    extracted_regions_with_min_data = get_data(extracted_regions_with_min)
    n_labels_with_min = len(np.unique(extracted_regions_with_min_data))

    assert n_labels_without_min > n_labels_with_min


def test_connected_label_regions_connect_diag_false(labels_img):
    labels_data = get_data(labels_img)
    n_labels_without_region_extraction = len(np.unique(labels_data))

    ext_reg_without_connect_diag = connected_label_regions(
        labels_img, connect_diag=False
    )

    data_wo_connect_diag = get_data(ext_reg_without_connect_diag)
    n_labels_wo_connect_diag = len(np.unique(data_wo_connect_diag))
    assert n_labels_wo_connect_diag > n_labels_without_region_extraction


def test_connected_label_regions_return_empty_for_large_min_size(labels_img):
    """If min_size is large and if all the regions are removed \
    then empty image will be returned.
    """
    extract_reg_min_size_large = connected_label_regions(
        labels_img, min_size=500
    )

    assert np.unique(get_data(extract_reg_min_size_large)) == 0


def test_connected_label_regions_check_labels(labels_img):
    """Test the names of the brain regions given in labels."""
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


def test_connected_label_regions_check_labels_as_numpy_array(labels_img):
    """Test the names of the brain regions given in labels."""
    # labels given in numpy array
    # Test labels for 9 regions in n_regions
    labels = [f"region_{x}" for x in "abcdefghi"]
    labels = np.asarray(labels)
    _, new_labels2 = connected_label_regions(labels_img, labels=labels)

    assert new_labels2 != ""
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


def test_connected_label_regions_unknonw_labels(
    labels_img, affine_eye, shape_3d_default
):
    """If unknown/negative integers are provided as labels in labels_img, \
    we raise an error and test the same whether error is raised.

    Introduce data type of float

    See issue: https://github.com/nilearn/nilearn/issues/2580
    """
    labels_data = get_data(labels_img)

    labels_data = np.zeros(shape_3d_default, dtype=np.float32)
    h0, h1, h2 = (x // 2 for x in shape_3d_default)
    labels_data[:h0, :h1, :h2] = 1
    labels_data[:h0, :h1, h2:] = 2
    labels_data[:h0, h1:, :h2] = 3
    labels_data[:h0, h1:, h2:] = -4
    labels_data[h0:, :h1, :h2] = 5
    labels_data[h0:, :h1, h2:] = 6
    labels_data[h0:, h1:, :h2] = np.nan
    labels_data[h0:, h1:, h2:] = np.inf

    neg_labels_img = Nifti1Image(labels_data, affine_eye)

    with pytest.raises(ValueError):
        connected_label_regions(labels_img=neg_labels_img)

    # If labels_img provided is 4D Nifti image, then test whether error is
    # raised or not. Since this function accepts only 3D image.
    labels_4d_data = np.zeros((*shape_3d_default, 2))
    labels_data[h0:, h1:, :h2] = 0
    labels_data[h0:, h1:, h2:] = 0
    labels_4d_data[..., 0] = labels_data
    labels_4d_data[..., 1] = labels_data
    labels_img_4d = Nifti1Image(labels_4d_data, affine_eye)

    with pytest.raises(DimensionError):
        connected_label_regions(labels_img=labels_img_4d)


def test_connected_label_regions_check_labels_string_without_list(
    labels_img, affine_eye, shape_3d_default
):
    """If labels (or names to regions) given is a string without a list \
    we expect it to be split to regions extracted and returned as list.
    """
    labels_in_str = "region_a"
    labels_img_in_str = generate_labeled_regions(
        shape=shape_3d_default, affine=affine_eye, n_regions=1
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
