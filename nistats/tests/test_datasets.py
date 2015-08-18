import os
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

import nibabel
from nilearn._utils.testing import (mock_request, wrap_chunk_read_,
                                    FetchFilesMock, assert_raises_regex)
from nilearn._utils.compat import _basestring
from nistats import datasets

tmpdir = None
file_mock = None
currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def test_fetch_localizer():
    dataset = datasets.fetch_localizer_first_level(data_dir=tmpdir)
    assert_true(isinstance(dataset.paradigm, _basestring))
    assert_true(isinstance(dataset.epi_img, _basestring))


def test_fetch_spm_auditory():
    dataset = datasets.fetch_spm_auditory(data_dir=tmpdir)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.func[0], _basestring))
    assert_equal(len(dataset.func), 96)


def test_fetch_spm_multimodal():
    dataset = datasets.fetch_spm_multimodal_fmri(data_dir=tmpdir)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.func1[0], _basestring))
    assert_equal(len(dataset.func1), 390)
    assert_true(isinstance(dataset.func2[0], _basestring))
    assert_equal(len(dataset.func2), 390)
    assert_equal(dataset.slice_order, 'descending')
    assert_true(dataset.trials_ses1, _basestring)
    assert_true(dataset.trials_ses2, _basestring)


def test_fiac():
    dataset = datasets.fetch_fiac_first_level(data_dir=tmpdir)
    assert_true(isinstance(dataset.func1, _basestring))
    assert_true(isinstance(dataset.func2, _basestring))
    assert_true(isinstance(dataset.design_matrix1, _basestring))
    assert_true(isinstance(dataset.design_matrix2, _basestring))
    assert_true(isinstance(dataset.mask, _basestring))
