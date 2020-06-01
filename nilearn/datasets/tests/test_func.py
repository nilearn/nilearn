"""
Test the datasets module
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import uuid

import numpy as np
import json
import nibabel
import gzip

import pandas as pd
import pytest
from nibabel.tmpdirs import TemporaryDirectory
from sklearn.utils import check_random_state

from . import test_utils as tst

from nilearn.datasets import utils, func
from nilearn._utils.testing import check_deprecation

from ..utils import _get_dataset_dir


@pytest.fixture()
def request_mocker():
    """ Mocks URL calls for functional dataset fetchers during testing.
    Tests the fetcher code without actually downloading the files.
    """
    tst.setup_mock(utils, func)
    yield
    tst.teardown_mock(utils, func)


@pytest.fixture()
def localizer_mocker():
    """ Mocks the index for localizer dataset.
    """
    setup_localizer()
    yield
    teardown_localizer()


with open(os.path.join(tst.datadir, 'localizer_index.json')) as of:
    localizer_template = json.load(of)
LOCALIZER_INDEX = {}
for idx in range(1, 95):
    idx = str(idx).zfill(2)
    sid = 'S{0}'.format(idx)
    LOCALIZER_INDEX.update(dict(
        (key.format(sid), uuid.uuid4().hex)
        for key in localizer_template))
LOCALIZER_INDEX['/localizer/phenotype/behavioural.tsv'] = uuid.uuid4().hex
LOCALIZER_PARTICIPANTS = np.recfromcsv(
    os.path.join(tst.datadir, 'localizer_participants.tsv'), delimiter='\t')
LOCALIZER_BEHAVIOURAL = np.recfromcsv(
    os.path.join(tst.datadir, 'localizer_behavioural.tsv'), delimiter='\t')


def mock_localizer_index(*args, **kwargs):
    return LOCALIZER_INDEX


def mock_np_recfromcsv(*args, **kwargs):
    if args[0].endswith('participants.tsv'):
        return LOCALIZER_PARTICIPANTS
    elif args[0].endswith('behavioural.tsv'):
        return LOCALIZER_BEHAVIOURAL
    else:
        raise ValueError('Unexpected args!')


def setup_localizer():
    global original_json_load
    global mock_json_load
    mock_json_load = mock_localizer_index
    original_json_load = json.load
    json.load = mock_json_load

    global original_np_recfromcsv
    global mock_np_recfromcsv
    mock_np_recfromcsv = mock_np_recfromcsv
    original_np_recfromcsv = np.recfromcsv
    np.recfromcsv = mock_np_recfromcsv


def teardown_localizer():
    global original_json_load
    json.load = original_json_load

    global original_np_recfromcsv
    np.recfromcsv = original_np_recfromcsv


def test_fetch_haxby(tmp_path, request_mocker):
    for i in range(1, 6):
        haxby = func.fetch_haxby(data_dir=str(tmp_path), subjects=[i],
                                 verbose=0)
        # subject_data + (md5 + mask if first subj)
        assert len(tst.mock_url_request.urls) == 1 + 2 * (i == 1)
        assert len(haxby.func) == 1
        assert len(haxby.anat) == 1
        assert len(haxby.session_target) == 1
        assert haxby.mask is not None
        assert len(haxby.mask_vt) == 1
        assert len(haxby.mask_face) == 1
        assert len(haxby.mask_house) == 1
        assert len(haxby.mask_face_little) == 1
        assert len(haxby.mask_house_little) == 1
        tst.mock_url_request.reset()
        assert haxby.description != ''

    # subjects with list
    subjects = [1, 2, 6]
    haxby = func.fetch_haxby(data_dir=str(tmp_path), subjects=subjects,
                             verbose=0)
    assert len(haxby.func) == len(subjects)
    assert len(haxby.mask_house_little) == len(subjects)
    assert len(haxby.anat) == len(subjects)
    assert haxby.anat[2] is None
    assert isinstance(haxby.mask, str)
    assert len(haxby.mask_face) == len(subjects)
    assert len(haxby.session_target) == len(subjects)
    assert len(haxby.mask_vt) == len(subjects)
    assert len(haxby.mask_face_little) == len(subjects)

    subjects = ['a', 8]
    message = "You provided invalid subject id {0} in a list"

    for sub_id in subjects:
        with pytest.raises(ValueError, match=message.format(sub_id)):
            func.fetch_haxby(data_dir=str(tmp_path), subjects=[sub_id])


def test_fetch_nyu_rest(tmp_path, request_mocker):
    # First session, all subjects
    with pytest.warns(np.VisibleDeprecationWarning,
                      match='fetch_nyu_rest has been deprecated'):
        nyu = func.fetch_nyu_rest(data_dir=str(tmp_path), verbose=0)
    assert len(tst.mock_url_request.urls) == 2
    assert len(nyu.func) == 25
    assert len(nyu.anat_anon) == 25
    assert len(nyu.anat_skull) == 25
    assert np.all(np.asarray(nyu.session) == 1)

    # All sessions, 12 subjects
    tst.mock_url_request.reset()
    with pytest.warns(np.VisibleDeprecationWarning,
                      match='fetch_nyu_rest has been deprecated'):
        nyu = func.fetch_nyu_rest(data_dir=str(tmp_path), 
                                  sessions=[1, 2, 3],
                                  n_subjects=12, verbose=0)
    # Session 1 has already been downloaded
    assert len(tst.mock_url_request.urls) == 2
    assert len(nyu.func) == 36
    assert len(nyu.anat_anon) == 36
    assert len(nyu.anat_skull) == 36
    s = np.asarray(nyu.session)
    assert np.all(s[:12] == 1)
    assert np.all(s[12:24] == 2)
    assert np.all(s[24:] == 3)
    assert nyu.description != ''


def test_fetch_adhd(tmp_path, request_mocker):
    local_url = "file://" + str(tmp_path / 'data')

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

    adhd = func.fetch_adhd(data_dir=str(tmp_path), url=local_url,
                           n_subjects=12, verbose=0)
    assert len(adhd.func) == 12
    assert len(adhd.confounds) == 12
    assert len(tst.mock_url_request.urls) == 13  # Subjects + phenotypic
    assert adhd.description != ''


def test_miyawaki2008(tmp_path, request_mocker):
    dataset = func.fetch_miyawaki2008(data_dir=str(tmp_path), verbose=0)
    assert len(dataset.func) == 32
    assert len(dataset.label) == 32
    assert isinstance(dataset.mask, str)
    assert len(dataset.mask_roi) == 38
    assert isinstance(dataset.background, str)
    assert len(tst.mock_url_request.urls) == 1
    assert dataset.description != ''


def test_fetch_localizer_contrasts(tmp_path, request_mocker, localizer_mocker):
    # 2 subjects
    dataset = func.fetch_localizer_contrasts(
        ['checkerboard'],
        n_subjects=2,
        data_dir=str(tmp_path),
        verbose=1)
    assert not hasattr(dataset, 'anats')
    assert not hasattr(dataset, 'tmaps')
    assert not hasattr(dataset, 'masks')
    assert isinstance(dataset.cmaps[0], str)
    assert isinstance(dataset.ext_vars, np.recarray)
    assert len(dataset.cmaps) == 2
    assert dataset.ext_vars.size == 2

    # Multiple contrasts
    dataset = func.fetch_localizer_contrasts(
        ['checkerboard', 'horizontal checkerboard'],
        n_subjects=2,
        data_dir=str(tmp_path),
        verbose=1)
    assert isinstance(dataset.ext_vars, np.recarray)
    assert isinstance(dataset.cmaps[0], str)
    assert len(dataset.cmaps) == 2 * 2  # two contrasts are fetched
    assert dataset.ext_vars.size == 2

    # all get_*=True
    dataset = func.fetch_localizer_contrasts(
        ['checkerboard'],
        n_subjects=1,
        data_dir=str(tmp_path),
        get_anats=True,
        get_masks=True,
        get_tmaps=True,
        verbose=1)
    assert isinstance(dataset.ext_vars, np.recarray)
    assert isinstance(dataset.anats[0], str)
    assert isinstance(dataset.cmaps[0], str)
    assert isinstance(dataset.masks[0], str)
    assert isinstance(dataset.tmaps[0], str)
    assert dataset.ext_vars.size == 1
    assert len(dataset.anats) == 1
    assert len(dataset.cmaps) == 1
    assert len(dataset.masks) == 1
    assert len(dataset.tmaps) == 1
    assert dataset.description != ''

    # grab a given list of subjects
    dataset2 = func.fetch_localizer_contrasts(
        ['checkerboard'],
        n_subjects=[2, 3, 5],
        data_dir=str(tmp_path),
        verbose=1)
    assert dataset2.ext_vars.size == 3
    assert len(dataset2.cmaps) == 3
    assert ([row[0] for row in dataset2.ext_vars] ==
                 [b'S02', b'S03', b'S05'])


def test_fetch_localizer_calculation_task(tmp_path, request_mocker,
                                          localizer_mocker):
    # 2 subjects
    dataset = func.fetch_localizer_calculation_task(
        n_subjects=2,
        data_dir=str(tmp_path),
        verbose=1)
    assert isinstance(dataset.ext_vars, np.recarray)
    assert isinstance(dataset.cmaps[0], str)
    assert dataset.ext_vars.size == 2
    assert len(dataset.cmaps) == 2
    assert dataset.description != ''


def test_fetch_localizer_button_task(tmp_path, request_mocker,
                                     localizer_mocker):
    local_url = "file://" + tst.datadir

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # Only one subject
    dataset = func.fetch_localizer_button_task(data_dir=str(tmp_path),
                                               url=local_url,
                                               verbose=1)

    assert isinstance(dataset.tmaps, list)
    assert isinstance(dataset.anats, list)

    assert len(dataset.tmaps) == 1
    assert len(dataset.anats) == 1

    assert isinstance(dataset.tmap, str)
    assert isinstance(dataset.anat, str)

    assert dataset.description != ''


def test_fetch_abide_pcp(tmp_path, request_mocker):
    local_url = "file://" + tst.datadir
    ids = [('50%03d' % i).encode() for i in range(800)]
    filenames = ['no_filename'] * 800
    filenames[::2] = ['filename'] * 400
    pheno = np.asarray(list(zip(ids, filenames)), dtype=[('subject_id', int),
                                                         ('FILE_ID', 'U11')])
    # pheno = pheno.T.view()
    tst.mock_fetch_files.add_csv('Phenotypic_V1_0b_preprocessed1.csv', pheno)

    # All subjects
    dataset = func.fetch_abide_pcp(data_dir=str(tmp_path), url=local_url,
                                   quality_checked=False, verbose=0)
    assert len(dataset.func_preproc) == 400
    assert dataset.description != ''

    # Smoke test using only a string, rather than a list of strings
    dataset = func.fetch_abide_pcp(data_dir=str(tmp_path), url=local_url,
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
        assert len(zmaps) == n_subjects * n_trials
        assert len(zmaps) == len(gain)


def test_fetch_mixed_gambles(tmp_path, request_mocker):
    local_url = "file://" + os.path.join(tst.datadir,
                                         "jimura_poldrack_2012_zmaps.zip")
    for n_subjects in [1, 5, 16]:
        mgambles = func.fetch_mixed_gambles(n_subjects=n_subjects,
                                            data_dir=str(tmp_path),
                                            url=local_url,
                                            verbose=0, return_raw_data=True)
        datasetdir = str(tmp_path / "jimura_poldrack_2012_zmaps")
        assert mgambles["zmaps"][0] == os.path.join(datasetdir, "zmaps",
                                                        "sub001_zmaps.nii.gz")
        assert len(mgambles["zmaps"]) == n_subjects


def test_check_parameters_megatrawls_datasets():
    # testing whether the function raises the same error message
    # if invalid input parameters are provided
    message = "Invalid {0} input is provided: {1}."

    for invalid_input_dim in [1, 5, 30]:
        with pytest.raises(
                ValueError,
                match=message.format('dimensionality', invalid_input_dim)):
            func.fetch_megatrawls_netmats(dimensionality=invalid_input_dim)

    for invalid_input_timeserie in ['asdf', 'time', 'st2']:
        with pytest.raises(
                ValueError,
                match=message.format('timeseries', invalid_input_timeserie)):
            func.fetch_megatrawls_netmats(timeseries=invalid_input_timeserie)

    for invalid_output_name in ['net1', 'net2']:
        with pytest.raises(
                ValueError,
                match=message.format('matrices', invalid_output_name)):
            func.fetch_megatrawls_netmats(matrices=invalid_output_name)


def test_fetch_megatrawls_netmats(tmp_path):
    # smoke test to see that files are fetched and read properly
    # since we are loading data present in it
    files_dir = str(tmp_path / 'Megatrawls'
                    / '3T_Q1-Q6related468_MSMsulc_d100_ts3')
    os.makedirs(files_dir)
    with open(os.path.join(files_dir, 'Znet2.txt'), 'w') as net_file:
        net_file.write("1")

    files_dir2 = str(tmp_path / 'Megatrawls'
                     / '3T_Q1-Q6related468_MSMsulc_d300_ts2')
    os.makedirs(files_dir2)
    with open(os.path.join(files_dir2, 'Znet1.txt'), 'w') as net_file2:
        net_file2.write("1")

    megatrawl_netmats_data = func.fetch_megatrawls_netmats(
        data_dir=str(tmp_path))

    # expected number of returns in output name should be equal
    assert len(megatrawl_netmats_data) == 5
    # check if returned bunch should not be empty
    # dimensions
    assert megatrawl_netmats_data.dimensions != ''
    # timeseries
    assert megatrawl_netmats_data.timeseries != ''
    # matrices
    assert megatrawl_netmats_data.matrices != ''
    # correlation matrices
    assert megatrawl_netmats_data.correlation_matrices != ''
    # description
    assert megatrawl_netmats_data.description != ''

    # check if input provided for dimensions, timeseries, matrices to be same
    # to user settings
    netmats_data = func.fetch_megatrawls_netmats(data_dir=str(tmp_path),
                                                 dimensionality=300,
                                                 timeseries='multiple_spatial_regression',
                                                 matrices='full_correlation')
    assert netmats_data.dimensions == 300
    assert netmats_data.timeseries == 'multiple_spatial_regression'
    assert netmats_data.matrices == 'full_correlation'


def test_fetch_cobre(tmp_path, request_mocker):
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
    cobre_dir = str(tmp_path / 'cobre')
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
    cobre_data = check_deprecation(
        func.fetch_cobre, "'fetch_cobre' has been deprecated")(
            n_subjects=None, data_dir=str(tmp_path), url=local_url)

    phenotypic_names = ['func', 'confounds', 'phenotypic', 'description',
                        'desc_con', 'desc_phenotypic']

    # test length of functional filenames to max 146
    assert len(cobre_data.func) == 146
    # test length of corresponding confounds files of same length to max 146
    assert len(cobre_data.confounds) == 146
    # test return type variables
    assert sorted(cobre_data) == sorted(phenotypic_names)
    # test functional filenames in a list
    assert isinstance(cobre_data.func, list)
    # test confounds files in a list
    assert isinstance(cobre_data.confounds, list)
    assert isinstance(cobre_data.func[0], str)
    # returned phenotypic data will be an array
    assert isinstance(cobre_data.phenotypic, np.recarray)

    # Fetch only 30 subjects
    data_30_subjects = func.fetch_cobre(n_subjects=30, url=local_url,
                                        data_dir=str(tmp_path))
    assert len(data_30_subjects.func) == 30
    assert len(data_30_subjects.confounds) == 30

    # Test more than maximum subjects
    test_150_subjects = func.fetch_cobre(n_subjects=150, url=local_url,
                                         data_dir=str(tmp_path))
    assert len(test_150_subjects.func) == 146
    os.remove(dummy)


def test_fetch_surf_nki_enhanced(tmp_path, request_mocker, verbose=0):

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

    nki_data = func.fetch_surf_nki_enhanced(data_dir=str(tmp_path),
                                            url=local_url)

    assert nki_data.description != ''
    assert len(nki_data.func_left) == 10
    assert len(nki_data.func_right) == 10
    assert isinstance(nki_data.phenotypic, np.ndarray)
    assert nki_data.phenotypic.shape == (10,)
    assert nki_data.description != ''


def _mock_participants_data(n_ids=5):
    """Maximum 8 ids are allowed to mock
    """
    ids = ['sub-pixar052', 'sub-pixar073', 'sub-pixar074', 'sub-pixar110',
           'sub-pixar042', 'sub-pixar109', 'sub-pixar068', 'sub-pixar007']
    array_ids = np.asarray(ids[:n_ids], dtype='|U12')

    age = np.ones(len(array_ids), dtype='<f8')
    age_group = np.asarray(len(array_ids) * ['2yo'], dtype='U3')
    child_adult = np.asarray(len(array_ids) * ['c'], dtype='U1')
    gender = np.asarray(len(array_ids) * ['m'], dtype='U1')
    handedness = np.asarray(len(array_ids) * ['r'], dtype='U1')
    csv = np.rec.array([array_ids, age, age_group, child_adult, gender,
                        handedness],
                       dtype=[('participant_id', '|U12'),
                              ('Age', '<f8'), ('AgeGroup', 'U3'),
                              ('Child_Adult', 'U1'), ('Gender', 'U1'),
                              ('Handedness', 'U1')])
    return csv


def test_fetch_development_fmri_participants(tmp_path, request_mocker):
    csv = _mock_participants_data()
    tst.mock_fetch_files.add_csv('participants.tsv', csv)
    local_url = 'file://' + os.path.join(tst.datadir)

    participants = func._fetch_development_fmri_participants(
        data_dir=str(tmp_path), url=local_url, verbose=1)
    assert isinstance(participants, np.ndarray)
    assert participants.shape == (5,)


def test_fetch_development_fmri_functional(tmp_path, request_mocker):
    csv = _mock_participants_data(n_ids=8)
    local_url = 'file://' + os.path.join(tst.datadir)
    funcs, confounds = func._fetch_development_fmri_functional(
        csv, data_dir=str(tmp_path), url=local_url, resume=True, verbose=1)
    assert len(funcs) == 8
    assert len(confounds) == 8


def test_fetch_development_fmri(tmp_path):
    data = func.fetch_development_fmri(n_subjects=2,
                                       data_dir=str(tmp_path), verbose=1)
    assert len(data.func) == 2
    assert len(data.confounds) == 2
    assert isinstance(data.phenotypic, np.ndarray)
    assert data.phenotypic.shape == (2,)
    assert data.description != ''

    # check reduced confounds
    confounds = np.recfromcsv(data.confounds[0], delimiter='\t')
    assert len(confounds[0]) == 15

    # check full confounds
    data = func.fetch_development_fmri(n_subjects=2, reduce_confounds=False,
                                       verbose=1)
    confounds = np.recfromcsv(data.confounds[0], delimiter='\t')
    assert len(confounds[0]) == 28

    # check first subject is an adult
    data = func.fetch_development_fmri(n_subjects=1, reduce_confounds=False,
                                       verbose=1)
    age_group = data.phenotypic['Child_Adult'][0]
    assert age_group == 'adult'

    # check first subject is an child if requested with age_group
    data = func.fetch_development_fmri(n_subjects=1, reduce_confounds=False,
                                       verbose=1, age_group='child')
    age_group = data.phenotypic['Child_Adult'][0]
    assert age_group == 'child'

    # check one of each age group returned if n_subject == 2
    # and age_group == 'both
    data = func.fetch_development_fmri(n_subjects=2, reduce_confounds=False,
                                       verbose=1, age_group='both')
    age_group = data.phenotypic['Child_Adult']
    assert(all(age_group == ['adult', 'child']))

    # check age_group
    data = func.fetch_development_fmri(n_subjects=2, reduce_confounds=False,
                                       verbose=1, age_group='child')
    assert(all([x == 'child' for x in data.phenotypic['Child_Adult']]))


def test_fetch_development_fmri_invalid_n_subjects():
    max_subjects = 155
    n_subjects = func._set_invalid_n_subjects_to_max(n_subjects=None,
                                                     max_subjects=max_subjects,
                                                     age_group='adult')
    assert n_subjects == max_subjects
    with pytest.warns(UserWarning, match='Wrong value for n_subjects='):
        func._set_invalid_n_subjects_to_max(n_subjects=-1,
                                            max_subjects=max_subjects,
                                            age_group='adult')


def test_fetch_development_fmri_exception():
    with pytest.raises(ValueError, match='Wrong value for age_group'):
        func._filter_func_regressors_by_participants(participants='junk',
                                                     age_group='junk for test')


# datasets tests originally belonging to nistats follow

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def test_fetch_bids_langloc_dataset(request_mocker, tmp_path):
    data_dir = str(tmp_path / 'bids_langloc_example')
    os.mkdir(data_dir)
    main_folder = os.path.join(data_dir, 'bids_langloc_dataset')
    os.mkdir(main_folder)

    datadir, dl_files = func.fetch_bids_langloc_dataset(str(tmp_path))

    assert isinstance(datadir, str)
    assert isinstance(dl_files, list)


def test_select_from_index():
    dataset_version = 'ds000030_R1.0.4'
    data_prefix = '{}/{}/uncompressed'.format(
        dataset_version.split('_')[0], dataset_version)
    # Prepare url files for subject and filter tests
    urls = [data_prefix + '/stuff.html',
            data_prefix + '/sub-xxx.html',
            data_prefix + '/sub-yyy.html',
            data_prefix + '/sub-xxx/ses-01_task-rest.txt',
            data_prefix + '/sub-xxx/ses-01_task-other.txt',
            data_prefix + '/sub-xxx/ses-02_task-rest.txt',
            data_prefix + '/sub-xxx/ses-02_task-other.txt',
            data_prefix + '/sub-yyy/ses-01.txt',
            data_prefix + '/sub-yyy/ses-02.txt']

    # Only 1 subject and not subject specific files get downloaded
    new_urls = func.select_from_index(urls, n_subjects=1)
    assert len(new_urls) == 6
    assert data_prefix + '/sub-yyy.html' not in new_urls

    # 2 subjects and not subject specific files get downloaded
    new_urls = func.select_from_index(urls, n_subjects=2)
    assert len(new_urls) == 9
    assert data_prefix + '/sub-yyy.html' in new_urls
    # ALL subjects and not subject specific files get downloaded
    new_urls = func.select_from_index(urls, n_subjects=None)
    assert len(new_urls) == 9

    # test inclusive filters. Only files with task-rest
    new_urls = func.select_from_index(
        urls, inclusion_filters=['*task-rest*'])
    assert len(new_urls) == 2
    assert data_prefix + '/stuff.html' not in new_urls

    # test exclusive filters. only files without ses-01
    new_urls = func.select_from_index(
        urls, exclusion_filters=['*ses-01*'])
    assert len(new_urls) == 6
    assert data_prefix + '/stuff.html' in new_urls

    # test filter combination. only files with task-rest and without ses-01
    new_urls = func.select_from_index(
        urls, inclusion_filters=['*task-rest*'],
        exclusion_filters=['*ses-01*'])
    assert len(new_urls) == 1
    assert data_prefix + '/sub-xxx/ses-02_task-rest.txt' in new_urls


def test_fetch_openneuro_dataset_index():
    with TemporaryDirectory() as tmpdir:
        dataset_version = 'ds000030_R1.0.4'
        subdir_names = ['ds000030', 'ds000030_R1.0.4', 'uncompressed']
        tmp_list = []
        for subdir in subdir_names:
            tmp_list.append(subdir)
            subdirpath = os.path.join(tmpdir, *tmp_list)
            os.mkdir(subdirpath)

        filepath = os.path.join(subdirpath, 'urls.json')
        mock_json_content = ['junk1', 'junk2']
        with open(filepath, 'w') as f:
            json.dump(mock_json_content, f)
        urls_path, urls = func.fetch_openneuro_dataset_index(
            data_dir=tmpdir,
            dataset_version=dataset_version,
            verbose=1,
        )
        urls_path = urls_path.replace('/', os.sep)
        assert urls_path == filepath
        assert urls == mock_json_content


def test_fetch_openneuro_dataset(request_mocker, tmp_path):
    dataset_version = 'ds000030_R1.0.4'
    data_prefix = '{}/{}/uncompressed'.format(
        dataset_version.split('_')[0], dataset_version)
    data_dir = _get_dataset_dir(data_prefix, data_dir=str(tmp_path),
                                verbose=1)
    url_file = os.path.join(data_dir, 'urls.json')
    # Prepare url files for subject and filter tests
    urls = [data_prefix + '/stuff.html',
            data_prefix + '/sub-xxx.html',
            data_prefix + '/sub-yyy.html',
            data_prefix + '/sub-xxx/ses-01_task-rest.txt',
            data_prefix + '/sub-xxx/ses-01_task-other.txt',
            data_prefix + '/sub-xxx/ses-02_task-rest.txt',
            data_prefix + '/sub-xxx/ses-02_task-other.txt',
            data_prefix + '/sub-yyy/ses-01.txt',
            data_prefix + '/sub-yyy/ses-02.txt']
    json.dump(urls, open(url_file, 'w'))

    # Only 1 subject and not subject specific files get downloaded
    datadir, dl_files = func.fetch_openneuro_dataset(
        urls, str(tmp_path), dataset_version)
    assert isinstance(datadir, str)
    assert isinstance(dl_files, list)
    assert len(dl_files) == 9


def test_fetch_localizer(request_mocker):
    dataset = func.fetch_localizer_first_level()
    assert isinstance(dataset['events'], str)
    assert isinstance(dataset.epi_img, str)


def _mock_original_spm_auditory_events_file():
    expected_events_data = {
        'onset': [factor * 42.0 for factor in range(0, 16)],
        'duration': [42.0] * 16,
        'trial_type': ['rest', 'active'] * 8,
    }
    expected_events_data = pd.DataFrame(expected_events_data)
    expected_events_data_string = expected_events_data.to_csv(
        sep='\t',
        index=0,
        columns=['onset', 'duration', 'trial_type'],
    )
    return expected_events_data_string


def _mock_bids_compliant_spm_auditory_events_file():
    events_filepath = os.path.join(os.getcwd(), 'tests_events.tsv')
    func._make_events_file_spm_auditory_data(
        events_filepath=events_filepath)
    with open(events_filepath, 'r') as actual_events_file_obj:
        actual_events_data_string = actual_events_file_obj.read()
    return actual_events_data_string, events_filepath


def test_fetch_language_localizer_demo_dataset(request_mocker, tmp_path):
    data_dir = str(tmp_path)
    expected_data_dir, expected_files = _mock_language_localizer_demo_dataset(
        data_dir)
    actual_data_dir, actual_subdirs = func.fetch_language_localizer_demo_dataset(
        data_dir)
    assert actual_data_dir == expected_data_dir
    assert actual_subdirs == expected_files


def _mock_language_localizer_demo_dataset(data_dir):
    events_file_paths = []
    derivatives_file_paths = []
    dataset_name = 'fMRI-language-localizer-demo-dataset'
    filename_prefix = '_task-languagelocalizer_'
    for name_suffix in range(1, 10):
        subject_dir = 'sub-0{}'.format(name_suffix)
        subdir_path = os.path.join(data_dir,
                                   dataset_name,
                                   subject_dir,
                                   'func',
                                   )
        os.makedirs(subdir_path)
        file_name = '{}{}events.tsv'.format(subject_dir, filename_prefix)
        file_path = os.path.join(subdir_path, file_name)
        events_file_paths.append(file_path)
        with open(file_path, 'w') as f:
            pass
        subdir_path = os.path.join(data_dir,
                                   dataset_name,
                                   'derivatives',
                                   subject_dir,
                                   'func',
                                   )
        os.makedirs(subdir_path)
        filename_suffixes = ['desc-preproc_bold.json',
                             'desc-preproc_bold.nii.gz',
                             'desc-confounds_regressors.tsv',
                             ]
        for filename_suffix_ in filename_suffixes:
            file_name = '{}{}'.format(filename_prefix, filename_suffix_)
            file_path = os.path.join(subdir_path, file_name)
            derivatives_file_paths.append(file_path)
            with open(file_path, 'w') as f:
                f
    expected_data_dir = os.path.join(data_dir, dataset_name)
    expected_files = []
    expected_files.extend(events_file_paths)
    expected_files.extend(derivatives_file_paths)
    expected_files.sort()
    return expected_data_dir, expected_files


def test_make_spm_auditory_events_file():
    try:
        (
            actual_events_data_string,
            events_filepath,
        ) = _mock_bids_compliant_spm_auditory_events_file()
    finally:
        os.remove(events_filepath)
    expected_events_data_string = _mock_original_spm_auditory_events_file()

    replace_win_line_ends = (
        lambda text: text.replace('\r\n', '\n')
        if text.find('\r\n') != -1 else text
    )
    actual_events_data_string = replace_win_line_ends(
        actual_events_data_string)
    expected_events_data_string = replace_win_line_ends(
        expected_events_data_string)

    assert actual_events_data_string == expected_events_data_string


def test_fetch_spm_auditory(request_mocker, tmp_path):
    import nibabel as nib
    import shutil
    saf = ["fM00223/fM00223_%03i.img" % index for index in range(4, 100)]
    saf_ = ["fM00223/fM00223_%03i.hdr" % index for index in range(4, 100)]

    data_dir = str(tmp_path / 'spm_auditory')
    os.mkdir(data_dir)
    subject_dir = os.path.join(data_dir, 'sub001')
    os.mkdir(subject_dir)
    os.mkdir(os.path.join(subject_dir, 'fM00223'))
    os.mkdir(os.path.join(subject_dir, 'sM00223'))

    path_img = str(tmp_path / 'tmp.img')
    path_hdr = str(tmp_path / 'tmp.hdr')
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4)), path_img)
    shutil.copy(path_img, os.path.join(subject_dir,
                                       "sM00223/sM00223_002.img"))
    shutil.copy(path_hdr, os.path.join(subject_dir,
                                       "sM00223/sM00223_002.hdr"))
    for file_ in saf:
        shutil.copy(path_img, os.path.join(subject_dir, file_))
    for file_ in saf_:
        shutil.copy(path_hdr, os.path.join(subject_dir, file_))

    dataset = func.fetch_spm_auditory(data_dir=str(tmp_path))
    assert isinstance(dataset.anat, str)
    assert isinstance(dataset.func[0], str)
    assert len(dataset.func) == 96


def test_fetch_spm_multimodal(request_mocker, tmp_path):
    data_dir = str(tmp_path / 'spm_multimodal_fmri')
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

    dataset = func.fetch_spm_multimodal_fmri(data_dir=str(tmp_path))
    assert isinstance(dataset.anat, str)
    assert isinstance(dataset.func1[0], str)
    assert len(dataset.func1) == 390
    assert isinstance(dataset.func2[0], str)
    assert len(dataset.func2) == 390
    assert dataset.slice_order == 'descending'
    assert isinstance(dataset.trials_ses1, str)
    assert isinstance(dataset.trials_ses2, str)


def test_fiac(request_mocker, tmp_path):
    # Create dummy 'files'
    fiac_dir = str(tmp_path / 'fiac_nilearn_stats' / 'nipy-data-0.2' /
                   'data' / 'fiac')
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

    dataset = func.fetch_fiac_first_level(data_dir=str(tmp_path))
    assert isinstance(dataset.func1, str)
    assert isinstance(dataset.func2, str)
    assert isinstance(dataset.design_matrix1, str)
    assert isinstance(dataset.design_matrix2, str)
    assert isinstance(dataset.mask, str)
