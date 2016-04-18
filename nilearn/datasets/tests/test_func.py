"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import numpy as np
import json
import nibabel
from sklearn.utils import check_random_state

from nose import with_setup
from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_not_equal)
from . import test_utils as tst

from nilearn.datasets import utils, func
from nilearn._utils.testing import assert_raises_regex

from nilearn._utils.compat import _basestring, _urllib


def setup_mock():
    return tst.setup_mock(utils, func)


def teardown_mock():
    return tst.teardown_mock(utils, func)


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_haxby_simple():
    local_url = "file:" + _urllib.request.pathname2url(os.path.join(tst.datadir,
        "pymvpa-exampledata.tar.bz2"))
    haxby = func.fetch_haxby_simple(data_dir=tst.tmpdir, url=local_url,
                                    verbose=0)
    datasetdir = os.path.join(tst.tmpdir, 'haxby2001_simple', 'pymvpa-exampledata')
    for key, file in [
            ('session_target', 'attributes.txt'),
            ('func', 'bold.nii.gz'),
            ('conditions_target', 'attributes_literal.txt')]:
        assert_equal(haxby[key], [os.path.join(datasetdir, file)])
        assert_true(os.path.exists(os.path.join(datasetdir, file)))

    assert_equal(haxby['mask'], os.path.join(datasetdir, 'mask.nii.gz'))


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fail_fetch_haxby_simple():
    # Test a dataset fetching failure to validate sandboxing
    local_url = "file:" + _urllib.request.pathname2url(os.path.join(tst.datadir,
        "pymvpa-exampledata.tar.bz2"))
    datasetdir = os.path.join(tst.tmpdir, 'haxby2001_simple', 'pymvpa-exampledata')
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
                  os.path.join(tst.tmpdir, 'haxby2001_simple'), files,
                  verbose=0)
    dummy = open(os.path.join(datasetdir, 'attributes.txt'), 'r')
    stuff = dummy.read(5)
    dummy.close()
    assert_equal(stuff, 'stuff')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_haxby():
    for i in range(1, 6):
        haxby = func.fetch_haxby(data_dir=tst.tmpdir, n_subjects=i,
                                 verbose=0)
        # subject_data + (md5 + mask if first subj)
        assert_equal(len(tst.mock_url_request.urls), 1 + 2 * (i == 1))
        assert_equal(len(haxby.func), i)
        assert_equal(len(haxby.anat), i)
        assert_equal(len(haxby.session_target), i)
        assert_true(haxby.mask is not None)
        assert_equal(len(haxby.mask_vt), i)
        assert_equal(len(haxby.mask_face), i)
        assert_equal(len(haxby.mask_house), i)
        assert_equal(len(haxby.mask_face_little), i)
        assert_equal(len(haxby.mask_house_little), i)
        tst.mock_url_request.reset()
        assert_not_equal(haxby.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_nyu_rest():
    # First session, all subjects
    nyu = func.fetch_nyu_rest(data_dir=tst.tmpdir, verbose=0)
    assert_equal(len(tst.mock_url_request.urls), 2)
    assert_equal(len(nyu.func), 25)
    assert_equal(len(nyu.anat_anon), 25)
    assert_equal(len(nyu.anat_skull), 25)
    assert_true(np.all(np.asarray(nyu.session) == 1))

    # All sessions, 12 subjects
    tst.mock_url_request.reset()
    nyu = func.fetch_nyu_rest(data_dir=tst.tmpdir, sessions=[1, 2, 3],
                              n_subjects=12, verbose=0)
    # Session 1 has already been downloaded
    assert_equal(len(tst.mock_url_request.urls), 2)
    assert_equal(len(nyu.func), 36)
    assert_equal(len(nyu.anat_anon), 36)
    assert_equal(len(nyu.anat_skull), 36)
    s = np.asarray(nyu.session)
    assert_true(np.all(s[:12] == 1))
    assert_true(np.all(s[12:24] == 2))
    assert_true(np.all(s[24:] == 3))
    assert_not_equal(nyu.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_adhd():
    local_url = "file://" + tst.datadir

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
    tst.mock_fetch_files.add_csv(
        'ADHD200_40subs_motion_parameters_and_phenotypics.csv',
        subs)

    adhd = func.fetch_adhd(data_dir=tst.tmpdir, url=local_url,
                           n_subjects=12, verbose=0)
    assert_equal(len(adhd.func), 12)
    assert_equal(len(adhd.confounds), 12)
    assert_equal(len(tst.mock_url_request.urls), 13)  # Subjects + phenotypic
    assert_not_equal(adhd.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_miyawaki2008():
    dataset = func.fetch_miyawaki2008(data_dir=tst.tmpdir, verbose=0)
    assert_equal(len(dataset.func), 32)
    assert_equal(len(dataset.label), 32)
    assert_true(isinstance(dataset.mask, _basestring))
    assert_equal(len(dataset.mask_roi), 38)
    assert_true(isinstance(dataset.background, _basestring))
    assert_equal(len(tst.mock_url_request.urls), 1)
    assert_not_equal(dataset.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_localizer_contrasts():
    local_url = "file://" + tst.datadir
    ids = np.asarray([('S%2d' % i).encode() for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    tst.mock_fetch_files.add_csv('cubicwebexport.csv', ids)
    tst.mock_fetch_files.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = func.fetch_localizer_contrasts(["checkerboard"],
                                             data_dir=tst.tmpdir,
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
                                             data_dir=tst.tmpdir,
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
        n_subjects=20, data_dir=tst.tmpdir,
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
                                             data_dir=tst.tmpdir,
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
                                             data_dir=tst.tmpdir,
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
                                             data_dir=tst.tmpdir,
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
                                             data_dir=tst.tmpdir,
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
    assert_not_equal(dataset.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_localizer_calculation_task():
    local_url = "file://" + tst.datadir
    ids = np.asarray(['S%2d' % i for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    tst.mock_fetch_files.add_csv('cubicwebexport.csv', ids)
    tst.mock_fetch_files.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = func.fetch_localizer_calculation_task(data_dir=tst.tmpdir,
                                                    url=local_url,
                                                    verbose=0)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 94)
    assert_equal(len(dataset.cmaps), 94)

    # 20 subjects
    dataset = func.fetch_localizer_calculation_task(n_subjects=20,
                                                    data_dir=tst.tmpdir,
                                                    url=local_url,
                                                    verbose=0)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 20)
    assert_equal(len(dataset.cmaps), 20)
    assert_not_equal(dataset.description, '')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_abide_pcp():
    local_url = "file://" + tst.datadir
    ids = [('50%03d' % i).encode() for i in range(800)]
    filenames = ['no_filename'] * 800
    filenames[::2] = ['filename'] * 400
    pheno = np.asarray(list(zip(ids, filenames)), dtype=[('subject_id', int),
                                                         ('FILE_ID', 'U11')])
    # pheno = pheno.T.view()
    tst.mock_fetch_files.add_csv('Phenotypic_V1_0b_preprocessed1.csv', pheno)

    # All subjects
    dataset = func.fetch_abide_pcp(data_dir=tst.tmpdir, url=local_url,
                                   quality_checked=False, verbose=0)
    assert_equal(len(dataset.func_preproc), 400)
    assert_not_equal(dataset.description, '')

    # Smoke test using only a string, rather than a list of strings
    dataset = func.fetch_abide_pcp(data_dir=tst.tmpdir, url=local_url,
                                   quality_checked=False, verbose=0,
                                   derivatives='func_preproc')

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
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_mixed_gambles():
    local_url = "file://" + os.path.join(tst.datadir,
                                         "jimura_poldrack_2012_zmaps.zip")
    for n_subjects in [1, 5, 16]:
        mgambles = func.fetch_mixed_gambles(n_subjects=n_subjects,
                                            data_dir=tst.tmpdir, url=local_url,
                                            verbose=0, return_raw_data=True)
        datasetdir = os.path.join(tst.tmpdir, "jimura_poldrack_2012_zmaps")
        assert_equal(mgambles["zmaps"][0], os.path.join(datasetdir, "zmaps",
                                                        "sub001_zmaps.nii.gz"))
        assert_equal(len(mgambles["zmaps"]), n_subjects)


def test_check_parameters_megatrawls_datasets():
    # testing whether the function raises the same error message
    # if invalid input parameters are provided
    message = "Invalid {0} input is provided: {1}."

    for invalid_input_dim in [1, 5, 30]:
        assert_raises_regex(ValueError,
                            message.format('dimensionality', invalid_input_dim),
                            func.fetch_megatrawls_netmats,
                            dimensionality=invalid_input_dim)

    for invalid_input_timeserie in ['asdf', 'time', 'st2']:
        assert_raises_regex(ValueError,
                            message.format('timeseries', invalid_input_timeserie),
                            func.fetch_megatrawls_netmats,
                            timeseries=invalid_input_timeserie)

    for invalid_output_name in ['net1', 'net2']:
        assert_raises_regex(ValueError,
                            message.format('matrices', invalid_output_name),
                            func.fetch_megatrawls_netmats,
                            matrices=invalid_output_name)


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_megatrawls_netmats():
    # smoke test to see that files are fetched and read properly
    # since we are loading data present in it
    files_dir = os.path.join(tst.tmpdir, 'Megatrawls', '3T_Q1-Q6related468_MSMsulc_d100_ts3')
    os.makedirs(files_dir)
    with open(os.path.join(files_dir, 'Znet2.txt'), 'w') as net_file:
        net_file.write("1")

    files_dir2 = os.path.join(tst.tmpdir, 'Megatrawls', '3T_Q1-Q6related468_MSMsulc_d300_ts2')
    os.makedirs(files_dir2)
    with open(os.path.join(files_dir2, 'Znet1.txt'), 'w') as net_file2:
        net_file2.write("1")

    megatrawl_netmats_data = func.fetch_megatrawls_netmats(data_dir=tst.tmpdir)

    # expected number of returns in output name should be equal
    assert_equal(len(megatrawl_netmats_data), 5)
    # check if returned bunch should not be empty
    # dimensions
    assert_not_equal(megatrawl_netmats_data.dimensions, '')
    # timeseries
    assert_not_equal(megatrawl_netmats_data.timeseries, '')
    # matrices
    assert_not_equal(megatrawl_netmats_data.matrices, '')
    # correlation matrices
    assert_not_equal(megatrawl_netmats_data.correlation_matrices, '')
    # description
    assert_not_equal(megatrawl_netmats_data.description, '')

    # check if input provided for dimensions, timeseries, matrices to be same
    # to user settings
    netmats_data = func.fetch_megatrawls_netmats(data_dir=tst.tmpdir,
                                                 dimensionality=300,
                                                 timeseries='multiple_spatial_regression',
                                                 matrices='full_correlation')
    assert_equal(netmats_data.dimensions, 300)
    assert_equal(netmats_data.timeseries, 'multiple_spatial_regression')
    assert_equal(netmats_data.matrices, 'full_correlation')


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_cobre():
    ids_sc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 21, 22, 25,
              28, 29, 32, 34, 37, 39, 40, 41, 42, 44, 46, 47, 49, 59, 60,
              64, 71, 72, 73, 75, 77, 78, 79, 80, 81, 82, 84, 85, 88, 89,
              92, 94, 96, 97, 98, 99, 100, 101, 103, 105, 106, 108, 109, 110,
              112, 117, 122, 126, 132, 133, 137, 142, 143, 145]
    ids_con = [13, 14, 17, 18, 19, 20, 23, 24, 26, 27, 30, 31, 33, 35, 36,
               38, 43, 45, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62,
               63, 65, 66, 67, 68, 69, 74, 76, 86, 87, 90, 91, 93, 95, 102,
               104, 107, 111, 113, 114, 115, 116, 118, 119, 120, 121, 123,
               124, 125, 127, 128, 129, 130, 131, 134, 135, 136, 138, 139,
               140, 141, 144, 146, 147]
    ids_sch = ['szxxx0040%03d' % i for i in ids_sc]
    ids_cont = ['contxxx0040%03d' % i for i in ids_con]
    ids = np.asarray(ids_sch + ids_cont, dtype='|U17')
    sz = np.asarray([i.startswith('s') for i in ids], dtype='<f8')
    age = np.ones(len(ids), dtype='<f8')
    sex = np.ones(len(ids), dtype='<f8')
    fd = np.ones(len(ids), dtype='<f8')
    csv = np.rec.array([ids, sz, age, sex, fd],
                       dtype=[('id', '|U17'), ('sz', '<f8'),
                              ('age', '<f8'), ('sex', '<f8'),
                              ('fd', '<f8')])
    tst.mock_fetch_files.add_csv('cobre_model_group.csv', csv)

    # Create a dummy 'files'
    cobre_dir = os.path.join(tst.tmpdir, 'cobre')
    os.mkdir(cobre_dir)
    dummy = os.path.join(cobre_dir, 'files')
    dummy_data = []
    for i in np.hstack([ids_sch, ids_cont]):
        # Func file
        f = 'fmri_' + i + '_session1_run1.nii.gz'
        m = 'fmri_' + i + '_session1_run1_extra.mat'
        dummy_data.append({'downloadUrl': 'whatever', 'name': f})
        dummy_data.append({'downloadUrl': 'whatever', 'name': m})

    # Add the CSV file
    dummy_data.append({
        'downloadUrl': 'whatever', 'name': 'cobre_model_group.csv'})
    json.dump(dummy_data, open(dummy, 'w'))
    local_url = "file://" + dummy

    # All subjects
    cobre_data = func.fetch_cobre(n_subjects=None, data_dir=tst.tmpdir,
                                  url=local_url)

    phenotypic_names = ['description', 'func', 'mat_files', 'phenotypic']
    # test length of functional filenames to max 146
    assert_equal(len(cobre_data.func), 146)
    # test length of corresponding matlab files of same length to max 146
    assert_equal(len(cobre_data.mat_files), 146)
    # test return type variables
    assert_equal(sorted(cobre_data), phenotypic_names)
    # test functional filenames in a list
    assert_true(isinstance(cobre_data.func, list))
    # test matlab files in a list
    assert_true(isinstance(cobre_data.mat_files, list))

    assert_true(isinstance(cobre_data.func[0], _basestring))
    # returned phenotypic data will be an array
    assert_true(isinstance(cobre_data.phenotypic, np.recarray))
    # data description should not be empty
    assert_not_equal(cobre_data.description, '')

    # Fetch only 30 subjects
    data_30_subjects = func.fetch_cobre(n_subjects=30, url=local_url,
                                        data_dir=tst.tmpdir)
    assert_equal(len(data_30_subjects.func), 30)
    assert_equal(len(data_30_subjects.mat_files), 30)

    # Test more than maximum subjects
    test_150_subjects = func.fetch_cobre(n_subjects=150, url=local_url,
                                         data_dir=tst.tmpdir)
    assert_equal(len(test_150_subjects.func), 146)
    os.remove(dummy)
