"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import numpy as np
import json
import nibabel
import gzip
from sklearn.utils import check_random_state

from nose import with_setup
from nose.tools import assert_true, assert_equal, assert_not_equal
from . import test_utils as tst

from nilearn.datasets import utils, func
from nilearn._utils.testing import assert_raises_regex

from nilearn._utils.compat import _basestring


def setup_mock():
    return tst.setup_mock(utils, func)


def teardown_mock():
    return tst.teardown_mock(utils, func)


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_haxby():
    for i in range(1, 6):
        haxby = func.fetch_haxby(data_dir=tst.tmpdir, subjects=[i],
                                 verbose=0)
        # subject_data + (md5 + mask if first subj)
        assert_equal(len(tst.mock_url_request.urls), 1 + 2 * (i == 1))
        assert_equal(len(haxby.func), 1)
        assert_equal(len(haxby.anat), 1)
        assert_equal(len(haxby.session_target), 1)
        assert_true(haxby.mask is not None)
        assert_equal(len(haxby.mask_vt), 1)
        assert_equal(len(haxby.mask_face), 1)
        assert_equal(len(haxby.mask_house), 1)
        assert_equal(len(haxby.mask_face_little), 1)
        assert_equal(len(haxby.mask_house_little), 1)
        tst.mock_url_request.reset()
        assert_not_equal(haxby.description, '')

    # subjects with list
    subjects = [1, 2, 6]
    haxby = func.fetch_haxby(data_dir=tst.tmpdir, subjects=subjects,
                             verbose=0)
    assert_equal(len(haxby.func), len(subjects))
    assert_equal(len(haxby.mask_house_little), len(subjects))
    assert_equal(len(haxby.anat), len(subjects))
    assert_true(haxby.anat[2] is None)
    assert_true(isinstance(haxby.mask, _basestring))
    assert_equal(len(haxby.mask_face), len(subjects))
    assert_equal(len(haxby.session_target), len(subjects))
    assert_equal(len(haxby.mask_vt), len(subjects))
    assert_equal(len(haxby.mask_face_little), len(subjects))

    subjects = ['a', 8]
    message = "You provided invalid subject id {0} in a list"

    for sub_id in subjects:
        assert_raises_regex(ValueError,
                            message.format(sub_id),
                            func.fetch_haxby,
                            data_dir=tst.tmpdir,
                            subjects=[sub_id])


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
            3154996, 3884955, 27034,
            4134561, 27018, 6115230,
            27037, 8409791, 27011]
    sub3 = [3007585, 8697774, 9750701,
            10064, 21019, 10042,
            10128, 2497695, 4164316,
            1552181, 4046678, 23012]
    sub4 = [1679142, 1206380, 23008,
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

    # grab a given list of subjects
    dataset2 = func.fetch_localizer_contrasts(["checkerboard"],
                                              n_subjects=[2, 3, 5],
                                              data_dir=tst.tmpdir,
                                              url=local_url,
                                              get_anats=True,
                                              get_masks=True,
                                              get_tmaps=True,
                                              verbose=0)

    # Check that we are getting only 3 subjects
    assert_equal(dataset2.ext_vars.size, 3)
    assert_equal(len(dataset2.anats), 3)
    assert_equal(len(dataset2.cmaps), 3)
    assert_equal(len(dataset2.masks), 3)
    assert_equal(len(dataset2.tmaps), 3)
    np.testing.assert_array_equal(dataset2.ext_vars,
                                  dataset.ext_vars[[1, 2, 4]])
    np.testing.assert_array_equal(dataset2.anats,
                                  np.array(dataset.anats)[[1, 2, 4]])
    np.testing.assert_array_equal(dataset2.cmaps,
                                  np.array(dataset.cmaps)[[1, 2, 4]])
    np.testing.assert_array_equal(dataset2.masks,
                                  np.array(dataset.masks)[[1, 2, 4]])
    np.testing.assert_array_equal(dataset2.tmaps,
                                  np.array(dataset.tmaps)[[1, 2, 4]])


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
    assert_equal(dataset.ext_vars.size, 1)
    assert_equal(len(dataset.cmaps), 1)

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
def test_fetch_localizer_button_task():
    local_url = "file://" + tst.datadir
    ids = np.asarray(['S%2d' % i for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    tst.mock_fetch_files.add_csv('cubicwebexport.csv', ids)
    tst.mock_fetch_files.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = func.fetch_localizer_button_task(data_dir=tst.tmpdir,
                                               url=local_url,
                                               verbose=0)
    assert_true(isinstance(dataset.ext_vars, np.recarray))
    assert_true(isinstance(dataset.cmaps[0], _basestring))
    assert_equal(dataset.ext_vars.size, 1)
    assert_equal(len(dataset.cmaps), 1)

    # 20 subjects
    dataset = func.fetch_localizer_button_task(n_subjects=20,
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
    ids_n = [40000, 40001, 40002, 40003, 40004, 40005, 40006, 40007, 40008,
             40009, 40010, 40011, 40012, 40013, 40014, 40015, 40016, 40017,
             40018, 40019, 40020, 40021, 40022, 40023, 40024, 40025, 40026,
             40027, 40028, 40029, 40030, 40031, 40032, 40033, 40034, 40035,
             40036, 40037, 40038, 40039, 40040, 40041, 40042, 40043, 40044,
             40045, 40046, 40047, 40048, 40049, 40050, 40051, 40052, 40053,
             40054, 40055, 40056, 40057, 40058, 40059, 40060, 40061, 40062,
             40063, 40064, 40065, 40066, 40067, 40068, 40069, 40071, 40072,
             40073, 40074, 40075, 40076, 40077, 40078, 40079, 40080, 40081,
             40082, 40084, 40085, 40086, 40087, 40088, 40089, 40090, 40091,
             40092, 40093, 40094, 40095, 40096, 40097, 40098, 40099, 40100,
             40101, 40102, 40103, 40104, 40105, 40106, 40107, 40108, 40109,
             40110, 40111, 40112, 40113, 40114, 40115, 40116, 40117, 40118,
             40119, 40120, 40121, 40122, 40123, 40124, 40125, 40126, 40127,
             40128, 40129, 40130, 40131, 40132, 40133, 40134, 40135, 40136,
             40137, 40138, 40139, 40140, 40141, 40142, 40143, 40144, 40145,
             40146, 40147]

    ids = np.asarray(ids_n, dtype='|U17')

    current_age = np.ones(len(ids), dtype='<f8')
    gender = np.ones(len(ids), dtype='<f8')
    handedness = np.ones(len(ids), dtype='<f8')

    subject_type = np.empty(len(ids), dtype="S10")
    subject_type[0:74] = 'Control'
    subject_type[74:146] = 'Patient'
    diagnosis = np.ones(len(ids), dtype='<f8')
    frames_ok = np.ones(len(ids), dtype='<f8')
    fd = np.ones(len(ids), dtype='<f8')
    fd_scrubbed = np.ones(len(ids), dtype='<f8')

    csv = np.rec.array([ids, current_age, gender, handedness, subject_type,
                        diagnosis, frames_ok, fd, fd_scrubbed],
                       dtype=[('ID', '|U17'), ('Current Age', '<f8'),
                              ('Gender', '<f8'), ('Handedness', '<f8'),
                              ('Subject Type', '|U17'), ('Diagnosis', '<f8'),
                              ('Frames OK', '<f8'), ('FD', '<f8'),
                              ('FD Scrubbed', '<f8')])

    # Create a dummy 'files'
    cobre_dir = os.path.join(tst.tmpdir, 'cobre')
    os.mkdir(cobre_dir)

    # Create the tsv
    name_f = os.path.join(cobre_dir, 'phenotypic_data.tsv')
    with open(name_f, 'wb') as f:
        header = '# {0}\n'.format('\t'.join(csv.dtype.names))
        f.write(header.encode())
        np.savetxt(f, csv, delimiter='\t', fmt='%s')

    # create an empty gz file
    f_in = open(name_f)
    name_f_gz = os.path.join(cobre_dir, 'phenotypic_data.tsv.gz')
    f_out = gzip.open(name_f_gz, 'wb')
    f_out.close()
    f_in.close()

    dummy = os.path.join(cobre_dir, '4197885')
    dummy_data = []

    for i in np.hstack(ids_n):
        # Func file
        f = 'fmri_00' + str(i) + '.nii.gz'

        m = 'fmri_00' + str(i) + '.tsv.gz'
        dummy_data.append({'download_url': 'whatever', 'name': f})
        dummy_data.append({'download_url': 'whatever', 'name': m})

    # Add the TSV file
    dummy_data.append({
        'download_url': 'whatever', 'name': 'phenotypic_data.tsv.gz'})
    # Add JSON files
    dummy_data.append({
        'download_url': 'whatever', 'name': 'keys_confounds.json'})
    dummy_data.append({
        'download_url': 'whatever', 'name': 'keys_phenotypic_data.json'})

    dummy_data = {'files': dummy_data}
    json.dump(dummy_data, open(dummy, 'w'))
    local_url = "file://" + dummy

    # All subjects
    cobre_data = func.fetch_cobre(n_subjects=None, data_dir=tst.tmpdir,
                                  url=local_url)

    phenotypic_names = ['func', 'confounds', 'phenotypic', 'description',
                        'desc_con', 'desc_phenotypic']

    # test length of functional filenames to max 146
    assert_equal(len(cobre_data.func), 146)
    # test length of corresponding confounds files of same length to max 146
    assert_equal(len(cobre_data.confounds), 146)
    # test return type variables
    assert_equal(sorted(cobre_data), sorted(phenotypic_names))
    # test functional filenames in a list
    assert_true(isinstance(cobre_data.func, list))
    # test confounds files in a list
    assert_true(isinstance(cobre_data.confounds, list))
    assert_true(isinstance(cobre_data.func[0], _basestring))
    # returned phenotypic data will be an array
    assert_true(isinstance(cobre_data.phenotypic, np.recarray))

    # Fetch only 30 subjects
    data_30_subjects = func.fetch_cobre(n_subjects=30, url=local_url,
                                        data_dir=tst.tmpdir)
    assert_equal(len(data_30_subjects.func), 30)
    assert_equal(len(data_30_subjects.confounds), 30)

    # Test more than maximum subjects
    test_150_subjects = func.fetch_cobre(n_subjects=150, url=local_url,
                                         data_dir=tst.tmpdir)
    assert_equal(len(test_150_subjects.func), 146)
    os.remove(dummy)


@with_setup(setup_mock, teardown_mock)
@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_fetch_surf_nki_enhanced(data_dir=tst.tmpdir, verbose=0):

    ids = np.asarray(['A00028185', 'A00035827', 'A00037511', 'A00039431',
                      'A00033747', 'A00035840', 'A00038998', 'A00035072',
                      'A00037112', 'A00039391'], dtype='U9')
    age = np.ones(len(ids), dtype='<f8')
    hand = np.asarray(len(ids) * ['x'], dtype='U1')
    sex = np.asarray(len(ids) * ['x'], dtype='U1')
    csv = np.rec.array([ids, age, hand, sex],
                       dtype=[('id', '|U19'), ('age', '<f8'),
                              ('hand', 'U1'), ('sex', 'U1')])

    tst.mock_fetch_files.add_csv('NKI_enhanced_surface_phenotypics.csv', csv)

    local_url = 'file://' + os.path.join(tst.datadir)

    nki_data = func.fetch_surf_nki_enhanced(data_dir=tst.tmpdir, url=local_url)

    assert_not_equal(nki_data.description, '')
    assert_equal(len(nki_data.func_left), 10)
    assert_equal(len(nki_data.func_right), 10)
    assert_true(isinstance(nki_data.phenotypic, np.ndarray))
    assert_equal(nki_data.phenotypic.shape, (10,))
    assert_not_equal(nki_data.description, '')
