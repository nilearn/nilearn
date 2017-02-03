"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
import nibabel
import numpy as np

from nose import with_setup
from nose.tools import assert_true, assert_equal, assert_not_equal
from . import test_utils as tst

from nilearn.datasets import utils, struct
from nilearn._utils.testing import assert_raises_regex

from nilearn._utils.compat import _basestring, get_header


def setup_mock():
    return tst.setup_mock(utils, struct)


def teardown_mock():
    return tst.teardown_mock(utils, struct)


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_get_dataset_dir():
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    expected_base_dir = os.path.expanduser('~/nilearn_data')
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tst.tmpdir, 'test_nilearn_data')
    os.environ['NILEARN_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tst.tmpdir, 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test', verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    expected_base_dir = os.path.join(tst.tmpdir, 'env_data')
    expected_dataset_dir = os.path.join(expected_base_dir, 'test')
    data_dir = utils._get_dataset_dir(
        'test', default_paths=[expected_dataset_dir], verbose=0)
    assert_equal(data_dir, os.path.join(expected_base_dir, 'test'))
    assert os.path.exists(data_dir)
    shutil.rmtree(data_dir)

    no_write = os.path.join(tst.tmpdir, 'no_write')
    os.makedirs(no_write)
    os.chmod(no_write, 0o400)

    expected_base_dir = os.path.join(tst.tmpdir, 'nilearn_shared_data')
    os.environ['NILEARN_SHARED_DATA'] = expected_base_dir
    data_dir = utils._get_dataset_dir('test',
                                      default_paths=[no_write],
                                      verbose=0)
    # Non writeable dir is returned because dataset may be in there.
    assert_equal(data_dir, no_write)
    assert os.path.exists(data_dir)
    os.chmod(no_write, 0o600)
    shutil.rmtree(data_dir)

    # Verify exception for a path which exists and is a file
    test_file = os.path.join(tst.tmpdir, 'some_file')
    with open(test_file, 'w') as out:
        out.write('abcfeg')
    assert_raises_regex(OSError,
                        'Nilearn tried to store the dataset '
                        'in the following directories, but',
                        utils._get_dataset_dir,
                        'test', test_file, verbose=0)


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_icbm152_2009():
    dataset = struct.fetch_icbm152_2009(data_dir=tst.tmpdir, verbose=0)
    assert_true(isinstance(dataset.csf, _basestring))
    assert_true(isinstance(dataset.eye_mask, _basestring))
    assert_true(isinstance(dataset.face_mask, _basestring))
    assert_true(isinstance(dataset.gm, _basestring))
    assert_true(isinstance(dataset.mask, _basestring))
    assert_true(isinstance(dataset.pd, _basestring))
    assert_true(isinstance(dataset.t1, _basestring))
    assert_true(isinstance(dataset.t2, _basestring))
    assert_true(isinstance(dataset.t2_relax, _basestring))
    assert_true(isinstance(dataset.wm, _basestring))
    assert_equal(len(tst.mock_url_request.urls), 1)
    assert_not_equal(dataset.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_oasis_vbm():
    local_url = "file://" + tst.datadir
    ids = np.asarray(['OAS1_%4d' % i for i in range(457)])
    ids = ids.view(dtype=[('ID', 'S9')])
    tst.mock_fetch_files.add_csv('oasis_cross-sectional.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    dataset = struct.fetch_oasis_vbm(data_dir=tst.tmpdir, url=local_url,
                                     verbose=0)
    assert_equal(len(dataset.gray_matter_maps), 403)
    assert_equal(len(dataset.white_matter_maps), 403)
    assert_true(isinstance(dataset.gray_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.white_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.data_usage_agreement, _basestring))
    assert_equal(len(tst.mock_url_request.urls), 3)

    dataset = struct.fetch_oasis_vbm(data_dir=tst.tmpdir, url=local_url,
                                     dartel_version=False, verbose=0)
    assert_equal(len(dataset.gray_matter_maps), 415)
    assert_equal(len(dataset.white_matter_maps), 415)
    assert_true(isinstance(dataset.gray_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.white_matter_maps[0], _basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.data_usage_agreement, _basestring))
    assert_equal(len(tst.mock_url_request.urls), 4)
    assert_not_equal(dataset.description, '')


def test_load_mni152_template():
    # All subjects
    template_nii = struct.load_mni152_template()
    assert_equal(template_nii.shape, (91, 109, 91))
    assert_equal(get_header(template_nii).get_zooms(), (2.0, 2.0, 2.0))


def test_load_mni152_brain_mask():
    brain_mask = struct.load_mni152_brain_mask()
    assert_true(isinstance(brain_mask, nibabel.Nifti1Image))
    # standard MNI template shape
    assert_equal(brain_mask.shape, (91, 109, 91))


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_icbm152_brain_gm_mask():
    dataset = struct.fetch_icbm152_2009(data_dir=tst.tmpdir, verbose=0)
    struct.load_mni152_template().to_filename(dataset.gm)
    grey_matter_img = struct.fetch_icbm152_brain_gm_mask(data_dir=tst.tmpdir,
                                                         verbose=0)
    assert_true(isinstance(grey_matter_img, nibabel.Nifti1Image))


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_surf_fsaverage():

    dataset = struct.fetch_surf_fsaverage5(data_dir=tst.tmpdir, verbose=0)

    keys = ['pial_left', 'pial_right', 'infl_left', 'infl_right',
            'sulc_left', 'sulc_right']

    filenames = ['pial.left.gii', 'pial.right.gii', 'pial_inflated.left.gii',
                 'pial_inflated.right.gii', 'sulc.left.gii', 'sulc.right.gii']

    for key, filename in zip(keys, filenames):
        assert_equal(dataset[key], os.path.join(tst.tmpdir, 'fsaverage5',
                                                filename))

    assert_not_equal(dataset.description, '')
    assert_equal(len(tst.mock_url_request.urls), len(keys))
