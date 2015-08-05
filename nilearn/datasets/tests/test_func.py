"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import shutil
import numpy as np
from tempfile import mkdtemp
import nibabel
from sklearn.utils import check_random_state

from nose import with_setup
from nose.tools import assert_true, assert_equal, assert_raises

from nilearn.datasets import utils, func
from nilearn._utils.testing import (mock_request, wrap_chunk_read_,
                                    FetchFilesMock)

from nilearn._utils.compat import _basestring


original_fetch_files = None
original_url_request = None
original_chunk_read = None
mock_fetch_files = None
mock_url_request = None
mock_chunk_read = None


def setup_mock():
    global original_url_request
    global mock_url_request
    mock_url_request = mock_request()
    original_url_request = utils._urllib.request
    utils._urllib.request = mock_url_request

    global original_chunk_read
    global mock_chunk_read
    mock_chunk_read = wrap_chunk_read_(utils._chunk_read_)
    original_chunk_read = utils._chunk_read_
    utils._chunk_read_ = mock_chunk_read

    global original_fetch_files
    global mock_fetch_files
    mock_fetch_files = FetchFilesMock()
    original_fetch_files = func._fetch_files
    func._fetch_files = mock_fetch_files


def teardown_mock():
    global original_url_request
    utils._urllib.request = original_url_request

    global original_chunk_read
    utils._chunk_read_ = original_chunk_read

    global original_fetch_files
    func._fetch_files = original_fetch_files


currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None


def setup_tmpdata():
    # create temporary dir
    global tmpdir
    tmpdir = mkdtemp()


def teardown_tmpdata():
    # remove temporary dir
    global tmpdir
    if tmpdir is not None:
        shutil.rmtree(tmpdir)


@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_haxby_simple():
    local_url = "file://" + os.path.join(datadir, "pymvpa-exampledata.tar.bz2")
    haxby = func.fetch_haxby_simple(data_dir=tmpdir, url=local_url,
                                    verbose=0)
    datasetdir = os.path.join(tmpdir, 'haxby2001_simple', 'pymvpa-exampledata')
    for key, file in [
            ('session_target', 'attributes.txt'),
            ('func', 'bold.nii.gz'),
            ('conditions_target', 'attributes_literal.txt')]:
        assert_equal(haxby[key], [os.path.join(datasetdir, file)])
        assert_true(os.path.exists(os.path.join(datasetdir, file)))

    assert_equal(haxby['mask'], os.path.join(datasetdir, 'mask.nii.gz'))


@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fail_fetch_haxby_simple():
    # Test a dataset fetching failure to validate sandboxing
    local_url = "file://" + os.path.join(datadir, "pymvpa-exampledata.tar.bz2")
    datasetdir = os.path.join(tmpdir, 'haxby2001_simple', 'pymvpa-exampledata')
    os.makedirs(datasetdir)
    # Create a dummy file. If sandboxing is successful, it won't be overwritten
    dummy = open(os.path.join(datasetdir, 'attributes.txt'), 'w')
    dummy.write('stuff')
    dummy.close()

    path = 'pymvpa-exampledata'

    opts = {'uncompress': True}
    files = [
        (os.path.join(path, 'attributes.txt'), local_url, opts),
        # The following file does not exists. It will cause an abortion of
        # the fetching procedure
        (os.path.join(path, 'bald.nii.gz'), local_url, opts)
    ]

    assert_raises(IOError, utils._fetch_files,
                  os.path.join(tmpdir, 'haxby2001_simple'), files,
                  verbose=0)
    dummy = open(os.path.join(datasetdir, 'attributes.txt'), 'r')
    stuff = dummy.read(5)
    dummy.close()
    assert_equal(stuff, 'stuff')


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_haxby():
    for i in range(1, 6):
        haxby = func.fetch_haxby(data_dir=tmpdir, n_subjects=i,
                                 verbose=0)
        # subject_data + (md5 + mask if first subj)
        assert_equal(len(mock_url_request.urls), 1 + 2 * (i == 1))
        assert_equal(len(haxby.func), i)
        assert_equal(len(haxby.anat), i)
        assert_equal(len(haxby.session_target), i)
        assert_true(haxby.mask is not None)
        assert_equal(len(haxby.mask_vt), i)
        assert_equal(len(haxby.mask_face), i)
        assert_equal(len(haxby.mask_house), i)
        assert_equal(len(haxby.mask_face_little), i)
        assert_equal(len(haxby.mask_house_little), i)
        mock_url_request.reset()


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_nyu_rest():
    # First session, all subjects
    nyu = func.fetch_nyu_rest(data_dir=tmpdir, verbose=0)
    assert_equal(len(mock_url_request.urls), 2)
    assert_equal(len(nyu.func), 25)
    assert_equal(len(nyu.anat_anon), 25)
    assert_equal(len(nyu.anat_skull), 25)
    assert_true(np.all(np.asarray(nyu.session) == 1))

    # All sessions, 12 subjects
    mock_url_request.reset()
    nyu = func.fetch_nyu_rest(data_dir=tmpdir, sessions=[1, 2, 3],
                              n_subjects=12, verbose=0)
    # Session 1 has already been downloaded
    assert_equal(len(mock_url_request.urls), 2)
    assert_equal(len(nyu.func), 36)
    assert_equal(len(nyu.anat_anon), 36)
    assert_equal(len(nyu.anat_skull), 36)
    s = np.asarray(nyu.session)
    assert_true(np.all(s[:12] == 1))
    assert_true(np.all(s[12:24] == 2))
    assert_true(np.all(s[24:] == 3))


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_adhd():
    local_url = "file://" + datadir

    sub1 = [3902469, 7774305, 3699991]
    sub2 = [2014113, 4275075, 1019436,
            3154996, 3884955,   27034,
            4134561,   27018, 6115230,
            27037, 8409791,   27011]
    sub3 = [3007585, 8697774, 9750701,
            10064,   21019,   10042,
            10128, 2497695, 4164316,
            1552181, 4046678,   23012]
    sub4 = [1679142, 1206380,   23008,
            4016887, 1418396, 2950754,
            3994098, 3520880, 1517058,
            9744150, 1562298, 3205761, 3624598]
    subs = np.array(sub1 + sub2 + sub3 + sub4, dtype='i8')
    subs = subs.view(dtype=[('Subject', 'i8')])
    mock_fetch_files.add_csv(
        'ADHD200_40subs_motion_parameters_and_phenotypics.csv',
        subs)

    adhd = func.fetch_adhd(data_dir=tmpdir, url=local_url,
                           n_subjects=12, verbose=0)
    assert_equal(len(adhd.func), 12)
    assert_equal(len(adhd.confounds), 12)
    assert_equal(len(mock_url_request.urls), 13)  # Subjects + phenotypic


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_miyawaki2008():
    dataset = func.fetch_miyawaki2008(data_dir=tmpdir, verbose=0)
    assert_equal(len(dataset.func), 32)
    assert_equal(len(dataset.label), 32)
    assert_true(isinstance(dataset.mask, _basestring))
    assert_equal(len(dataset.mask_roi), 38)
    assert_equal(len(mock_url_request.urls), 1)


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_localizer_contrasts():
    local_url = "file://" + datadir
    ids = np.asarray([('S%2d' % i).encode() for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    mock_fetch_files.add_csv('cubicwebexport.csv', ids)
    mock_fetch_files.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = func.fetch_localizer_contrasts(["checkerboard"],
                                             data_dir=tmpdir,
                                             url=local_url,
                                             verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)

    # 20 subjects
    dataset = func.fetch_localizer_contrasts(["checkerboard"],
                                             n_subjects=20,
                                             data_dir=tmpdir,
                                             url=local_url,
                                             verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_equal(len(dataset.cmaps), 20)
    assert_equal(dataset.ext_vars.size, 20)

    # Multiple contrasts
    dataset = func.fetch_localizer_contrasts(
        ["checkerboard", "horizontal checkerboard"],
        n_subjects=20, data_dir=tmpdir,
        verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(len(dataset.cmaps), 20 * 2)  # two contrasts are fetched
    assert_equal(dataset.ext_vars.size, 20)

    # get_anats=True
    dataset = func.fetch_localizer_contrasts(["checkerboard"],
                                             data_dir=tmpdir,
                                             url=local_url,
                                             get_anats=True,
                                             verbose=0)
    assert_true(dataset.masks is None)
    assert_true(dataset.tmaps is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.anats[0], _basestring))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.anats), 94)
    assert_equal(len(dataset.cmaps), 94)

    # get_masks=True
    dataset = func.fetch_localizer_contrasts(["checkerboard"],
                                             data_dir=tmpdir,
                                             url=local_url,
                                             get_masks=True,
                                             verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.tmaps is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_true(isinstance(dataset.masks[0], _basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)
    assert_equal(len(dataset.masks), 94)

    # get_tmaps=True
    dataset = func.fetch_localizer_contrasts(["checkerboard"],
                                             data_dir=tmpdir,
                                             url=local_url,
                                             get_tmaps=True,
                                             verbose=0)
    assert_true(dataset.anats is None)
    assert_true(dataset.masks is None)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_true(isinstance(dataset.tmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)
    assert_equal(len(dataset.tmaps), 94)

    # all get_*=True
    dataset = func.fetch_localizer_contrasts(["checkerboard"],
                                             data_dir=tmpdir,
                                             url=local_url,
                                             get_anats=True,
                                             get_masks=True,
                                             get_tmaps=True,
                                             verbose=0)

    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.anats[0], _basestring))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_true(isinstance(dataset.masks[0], _basestring))
    assert_true(isinstance(dataset.tmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.anats), 94)
    assert_equal(len(dataset.cmaps), 94)
    assert_equal(len(dataset.masks), 94)
    assert_equal(len(dataset.tmaps), 94)


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_localizer_calculation_task():
    local_url = "file://" + datadir
    ids = np.asarray(['S%2d' % i for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    mock_fetch_files.add_csv('cubicwebexport.csv', ids)
    mock_fetch_files.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = func.fetch_localizer_calculation_task(data_dir=tmpdir,
                                                    url=local_url,
                                                    verbose=0)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)

    # 20 subjects
    dataset = func.fetch_localizer_calculation_task(n_subjects=20,
                                                    data_dir=tmpdir,
                                                    url=local_url,
                                                    verbose=0)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 20)
    assert_equal(len(dataset.cmaps), 20)


@with_setup(setup_mock, teardown_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_abide_pcp():
    local_url = "file://" + datadir
    ids = [('50%03d' % i).encode() for i in range(800)]
    filenames = ['no_filename'] * 800
    filenames[::2] = ['filename'] * 400
    pheno = np.asarray(list(zip(ids, filenames)), dtype=[('subject_id', int),
                                                         ('FILE_ID', 'U11')])
    # pheno = pheno.T.view()
    mock_fetch_files.add_csv('Phenotypic_V1_0b_preprocessed1.csv', pheno)

    # All subjects
    dataset = func.fetch_abide_pcp(data_dir=tmpdir, url=local_url,
                                   quality_checked=False, verbose=0)
    assert_equal(len(dataset.func_preproc), 400)


def test__load_mixed_gambles():
    rng = check_random_state(42)
    n_trials = 48
    affine = np.eye(4)
    for n_subjects in [1, 5, 16]:
        zmaps = []
        for _ in range(n_subjects):
            zmaps.append(nibabel.Nifti1Image(rng.randn(3, 4, 5, n_trials),
                                             affine))
        zmaps, gain, _ = func._load_mixed_gambles(zmaps)
        assert_equal(len(zmaps), n_subjects * n_trials)
        assert_equal(len(zmaps), len(gain))


@with_setup(setup_mock)
@with_setup(setup_tmpdata, teardown_tmpdata)
def test_fetch_mixed_gambles():
    local_url = "file://" + os.path.join(datadir,
                                         "jimura_poldrack_2012_zmaps.zip")
    for n_subjects in [1, 5, 16]:
        mgambles = func.fetch_mixed_gambles(n_subjects=n_subjects,
                                            data_dir=tmpdir, url=local_url,
                                            verbose=0, return_raw_data=True)
        datasetdir = os.path.join(tmpdir, "jimura_poldrack_2012_zmaps/")
        assert_equal(mgambles["zmaps"][0], os.path.join(datasetdir, "zmaps",
                                                        "sub001_zmaps.nii.gz"))
        assert_equal(len(mgambles["zmaps"]), n_subjects)
