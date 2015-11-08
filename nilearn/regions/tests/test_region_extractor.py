""" Test Region Extractor and its functions """

import numpy as np
import nibabel

from nose.tools import assert_raises, assert_equal

from nilearn.regions.region_extractor import (connected_regions,
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


def test_validity_extract_types_in_connected_regions():
    shape = (91, 109, 91)
    n_regions = 2
    maps = generate_maps(shape, n_regions)
    map_0 = maps[0]
    valid_names = ['connected_components', 'local_regions']

    # test whether same error raises as expected when given invalid inputs to
    # extract_type of regions extraction
    message = ("'extract_type' should be {0}").format(valid_names)
    for img in iter_img(map_0):
        assert_raises_regex(ValueError,
                            message,
                            connected_regions,
                            img, min_size=40,
                            extract_type='connect_regions')

    # test whether error raises when there is an invalid inputs to extract
    invalid_extract_type = ['local', 'asdf', 10]
    for type_ in invalid_extract_type:
        assert_raises(ValueError, connected_regions,
                      img, extract_type=type_)


def test_passing_connected_regions():
    shape = (91, 109, 91)
    n_regions = 4
    maps = generate_maps(shape, n_regions)
    map_0 = maps[0]

    valid_extract_types = ['connected_components', 'local_regions']
    # smoke test for function connected_regions
    for map_, type_ in zip(iter_img(map_0), valid_extract_types):
        regions_extracted = connected_regions(
            map_, min_size=30, extract_type=type_, peak_local_smooth=3)


def test_passing_RegionExtractor_object():
    shape = (91, 109, 91)
    n_regions = 2
    n_subjects = 5
    maps = generate_maps(shape, n_regions)
    map_0 = maps[0]

    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(np.ones((shape), dtype=np.int8), affine)

    # smoke test fit() and fit_transform() with giving inputs
    extractor = RegionExtractor(map_0, threshold=0.5, mask_img=mask_img)
    # smoke test fit() function
    extractor.fit()
    n_regions = extractor.regions_.shape[-1]

    imgs = []
    signals = []
    shape = (91, 109, 91, 7)
    expected_signal_shape = (7, n_regions)
    for id_ in range(n_subjects):
        img, data = _make_random_data(shape)
        # smoke test NiftiMapsMasker transform inherited in Region Extractor
        signal = extractor.transform(img)
        assert_equal(expected_signal_shape, signal.shape)
