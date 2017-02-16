import os
import json
import zipfile
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

import nibabel
from nilearn._utils.testing import (mock_request, wrap_chunk_read_,
                                    FetchFilesMock, assert_raises_regex)
from nilearn.datasets.tests import test_utils as tst
from nilearn.datasets import utils, func
from nilearn._utils.compat import _basestring
from nose import with_setup

from nistats import datasets


currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def setup_mock():
    return tst.setup_mock(utils, func)


def teardown_mock():
    return tst.teardown_mock(utils, func)


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_bids_langloc_dataset():
    data_dir = os.path.join(tst.tmpdir, 'bids_langloc_example')
    os.mkdir(data_dir)
    main_folder = os.path.join(data_dir, 'bids_langloc_dataset')
    os.mkdir(main_folder)

    datadir, dl_files = datasets.fetch_bids_langloc_dataset(tst.tmpdir)

    assert_true(isinstance(datadir, _basestring))
    assert_true(isinstance(dl_files, list))


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_openfmri_dataset():
    # test dataset not found
    data_dir = os.path.join(tst.tmpdir, 'ds000001')
    os.mkdir(data_dir)
    api_content = [dict(accession_number='dsother')]
    json.dump(api_content, open(os.path.join(data_dir, 'api'), 'w'))
    assert_raises(ValueError, datasets.fetch_openfmri_dataset,
                  data_dir=tst.tmpdir)
    # test dataset found with no revision
    data_dir = os.path.join(tst.tmpdir, 'dsother')
    os.mkdir(data_dir)
    api_content = [dict(accession_number='dsother', revision_set=[],
                        link_set=[dict(revision=None, url='http')])]
    json.dump(api_content, open(os.path.join(data_dir, 'api'), 'w'))
    data_dir, dl_files = datasets.fetch_openfmri_dataset(
        dataset_name='dsother', data_dir=tst.tmpdir)
    assert_true(isinstance(data_dir, _basestring))
    assert_true(isinstance(dl_files, list))


@with_setup(setup_mock, teardown_mock)
def test_fetch_localizer():
    dataset = datasets.fetch_localizer_first_level()
    assert_true(isinstance(dataset.paradigm, _basestring))
    assert_true(isinstance(dataset.epi_img, _basestring))


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_spm_auditory():
    import nibabel as nib
    import shutil
    saf = ["fM00223/fM00223_%03i.img" % index for index in range(4, 100)]
    saf_ = ["fM00223/fM00223_%03i.hdr" % index for index in range(4, 100)]

    data_dir = os.path.join(tst.tmpdir, 'spm_auditory')
    os.mkdir(data_dir)
    subject_dir = os.path.join(data_dir, 'sub001')
    os.mkdir(subject_dir)
    os.mkdir(os.path.join(subject_dir, 'fM00223'))
    os.mkdir(os.path.join(subject_dir, 'sM00223'))

    path_img = os.path.join(tst.tmpdir, 'tmp.img')
    path_hdr = os.path.join(tst.tmpdir, 'tmp.hdr')
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4)), path_img)
    shutil.copy(path_img, os.path.join(subject_dir, "sM00223/sM00223_002.img"))
    shutil.copy(path_hdr, os.path.join(subject_dir, "sM00223/sM00223_002.hdr"))
    for file_ in saf:
        shutil.copy(path_img, os.path.join(subject_dir, file_))
    for file_ in saf_:
        shutil.copy(path_hdr, os.path.join(subject_dir, file_))

    dataset = datasets.fetch_spm_auditory(data_dir=tst.tmpdir)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.func[0], _basestring))
    assert_equal(len(dataset.func), 96)


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_spm_multimodal():
    data_dir = os.path.join(tst.tmpdir, 'spm_multimodal_fmri')
    os.mkdir(data_dir)
    subject_dir = os.path.join(data_dir, 'sub001')
    os.mkdir(subject_dir)
    os.mkdir(os.path.join(subject_dir, 'fMRI'))
    os.mkdir(os.path.join(subject_dir, 'sMRI'))
    open(os.path.join(subject_dir, 'sMRI', 'smri.img'), 'a').close()
    for session in [0, 1]:
        open(os.path.join(subject_dir, 'fMRI',
                          'trials_ses%i.mat' % (session + 1)), 'a').close()
        dir_ = os.path.join(subject_dir, 'fMRI', 'Session%d' % (session + 1))
        os.mkdir(dir_)
        for i in range(390):
            open(os.path.join(dir_, 'fMETHODS-000%i-%i-01.img' %
                              (session + 5, i)), 'a').close()

    dataset = datasets.fetch_spm_multimodal_fmri(data_dir=tst.tmpdir)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.func1[0], _basestring))
    assert_equal(len(dataset.func1), 390)
    assert_true(isinstance(dataset.func2[0], _basestring))
    assert_equal(len(dataset.func2), 390)
    assert_equal(dataset.slice_order, 'descending')
    assert_true(dataset.trials_ses1, _basestring)
    assert_true(dataset.trials_ses2, _basestring)


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fiac():
    # Create dummy 'files'
    fiac_dir = os.path.join(tst.tmpdir, 'fiac_nistats', 'nipy-data-0.2',
                            'data', 'fiac')
    fiac0_dir = os.path.join(fiac_dir, 'fiac0')
    os.makedirs(fiac0_dir)
    for session in [1, 2]:
        # glob func data for session session + 1
        session_func = os.path.join(fiac0_dir, 'run%i.nii.gz' % session)
        open(session_func, 'a').close()
        sess_dmtx = os.path.join(fiac0_dir, 'run%i_design.npz' % session)
        open(sess_dmtx, 'a').close()
    mask = os.path.join(fiac0_dir, 'mask.nii.gz')
    open(mask, 'a').close()

    dataset = datasets.fetch_fiac_first_level(data_dir=tst.tmpdir)
    assert_true(isinstance(dataset.func1, _basestring))
    assert_true(isinstance(dataset.func2, _basestring))
    assert_true(isinstance(dataset.design_matrix1, _basestring))
    assert_true(isinstance(dataset.design_matrix2, _basestring))
    assert_true(isinstance(dataset.mask, _basestring))
