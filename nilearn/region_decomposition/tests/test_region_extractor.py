""" Test Region Extractor and its functions """

import numpy as np
import nibabel

from nose.tools import assert_raises

from nilearn.region_decomposition.region_extractor import (apply_threshold_to_maps,
                                                           extract_regions,
                                                           RegionExtractor)
from nilearn.image import iter_img

from nilearn._utils.testing import assert_raises_regex, generate_maps
from nilearn.decomposition.tests.test_canica import _make_canica_components


def _make_random_data(shape):
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    data_rng = rng.normal(size=shape)
    img = nibabel.Nifti1Image(data_rng, affine)
    data = img.get_data()
    return img, data


def test_apply_threshold_to_maps():
    valid_threshold_value = [0.5, 'auto']
    invalid_threshold_value = ['10', 'some_value']
    valid_threshold_strategies = ['percentile', 'voxelratio']
    invalid_threshold_strategies = ['percent', 'ratio', 'auto']

    shape = (6, 8, 10)
    _, data = _make_random_data(shape)
    data = _make_canica_components(shape)

    for strategy_invalid in invalid_threshold_strategies:
        assert_raises(ValueError, apply_threshold_to_maps,
                      data, 'auto', strategy_invalid)

    for thr in invalid_threshold_value:
        assert_raises(ValueError, apply_threshold_to_maps,
                      data, thr, 'percentile')
    maps = generate_maps(shape, n_regions=1)
    map_0 = maps[0]
    # smoke test for function apply_threshold_to_maps with random data
    for map_, strategy in zip(iter_img(map_0), valid_threshold_strategies):
        data = map_.get_data()
        thr_maps = apply_threshold_to_maps(data, 'auto', strategy)


def test_extract_regions():
    valid_extract_types = ['auto', 'local_regions']
    invalid_extract_type = ['local', 'asdf', 10]
    shape = (91, 109, 91)
    n_regions = 2
    maps = generate_maps(shape, n_regions)
    map_0 = maps[0]

    assert_raises(ValueError, extract_regions, map_0, min_size=10,
                  extract_type='auto', smooth_fwhm=3)

    for type_ in invalid_extract_type:
        assert_raises(ValueError, extract_regions,
                      map_0, 20, type_, 5)

    # smoke test for function extract_regions
    for map_, type_ in zip(iter_img(map_0), valid_extract_types):
        regions_extracted = extract_regions(map_, min_size=30,
                                            extract_type=type_,
                                            smooth_fwhm=3)


def test_region_extractor_function():
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
