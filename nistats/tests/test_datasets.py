import json
import os

import numpy as np
import pandas as pd

from nibabel.tmpdirs import TemporaryDirectory
from nilearn._utils.compat import _basestring
from nilearn.datasets.utils import _get_dataset_dir

from nistats import datasets
from nistats.datasets import fetch_openneuro_dataset_index, \
    fetch_language_localizer_demo_dataset

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def test_fetch_bids_langloc_dataset(request_mocker, tmp_path):
    data_dir = str(tmp_path / 'bids_langloc_example')
    os.mkdir(data_dir)
    main_folder = os.path.join(data_dir, 'bids_langloc_dataset')
    os.mkdir(main_folder)

    datadir, dl_files = datasets.fetch_bids_langloc_dataset(str(tmp_path))

    assert isinstance(datadir, _basestring)
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
    new_urls = datasets.select_from_index(urls, n_subjects=1)
    assert len(new_urls) == 6
    assert data_prefix + '/sub-yyy.html' not in new_urls

    # 2 subjects and not subject specific files get downloaded
    new_urls = datasets.select_from_index(urls, n_subjects=2)
    assert len(new_urls) == 9
    assert data_prefix + '/sub-yyy.html' in new_urls
    # ALL subjects and not subject specific files get downloaded
    new_urls = datasets.select_from_index(urls, n_subjects=None)
    assert len(new_urls) == 9

    # test inclusive filters. Only files with task-rest
    new_urls = datasets.select_from_index(
        urls, inclusion_filters=['*task-rest*'])
    assert len(new_urls) == 2
    assert data_prefix + '/stuff.html' not in new_urls

    # test exclusive filters. only files without ses-01
    new_urls = datasets.select_from_index(
        urls, exclusion_filters=['*ses-01*'])
    assert len(new_urls) == 6
    assert data_prefix + '/stuff.html' in new_urls

    # test filter combination. only files with task-rest and without ses-01
    new_urls = datasets.select_from_index(
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
        urls_path, urls = fetch_openneuro_dataset_index(
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
    datadir, dl_files = datasets.fetch_openneuro_dataset(
        urls, str(tmp_path), dataset_version)
    assert isinstance(datadir, _basestring)
    assert isinstance(dl_files, list)
    assert len(dl_files) == 9


def test_fetch_localizer():
    dataset = datasets.fetch_localizer_first_level()
    assert isinstance(dataset['events'], _basestring)
    assert isinstance(dataset.epi_img, _basestring)
    

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
    datasets._make_events_file_spm_auditory_data(
        events_filepath=events_filepath)
    with open(events_filepath, 'r') as actual_events_file_obj:
        actual_events_data_string = actual_events_file_obj.read()
    return actual_events_data_string, events_filepath


def test_fetch_language_localizer_demo_dataset(request_mocker, tmp_path):
    data_dir = str(tmp_path)
    expected_data_dir, expected_files= _mock_language_localizer_demo_dataset(
            data_dir)
    actual_data_dir, actual_subdirs = fetch_language_localizer_demo_dataset(
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
        with open(file_path, 'w') as f: pass
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
            with open(file_path, 'w') as f: pass
    expected_data_dir = os.path.join(data_dir, dataset_name)
    expected_files = []
    expected_files.extend(events_file_paths)
    expected_files.extend(derivatives_file_paths)
    expected_files.sort()
    return expected_data_dir, expected_files


def test_make_spm_auditory_events_file():
    try:
        actual_events_data_string, events_filepath = _mock_bids_compliant_spm_auditory_events_file()
    finally:
        os.remove(events_filepath)
    expected_events_data_string = _mock_original_spm_auditory_events_file()
    
    replace_win_line_ends = (lambda text: text.replace('\r\n', '\n')
                                if text.find('\r\n') != -1 else text
                             )
    actual_events_data_string = replace_win_line_ends(actual_events_data_string)
    expected_events_data_string = replace_win_line_ends(expected_events_data_string)
    
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
    shutil.copy(path_img, os.path.join(subject_dir, "sM00223/sM00223_002.img"))
    shutil.copy(path_hdr, os.path.join(subject_dir, "sM00223/sM00223_002.hdr"))
    for file_ in saf:
        shutil.copy(path_img, os.path.join(subject_dir, file_))
    for file_ in saf_:
        shutil.copy(path_hdr, os.path.join(subject_dir, file_))

    dataset = datasets.fetch_spm_auditory(data_dir=str(tmp_path))
    assert isinstance(dataset.anat, _basestring)
    assert isinstance(dataset.func[0], _basestring)
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

    dataset = datasets.fetch_spm_multimodal_fmri(data_dir=str(tmp_path))
    assert isinstance(dataset.anat, _basestring)
    assert isinstance(dataset.func1[0], _basestring)
    assert len(dataset.func1) == 390
    assert isinstance(dataset.func2[0], _basestring)
    assert len(dataset.func2) == 390
    assert dataset.slice_order == 'descending'
    assert isinstance(dataset.trials_ses1, _basestring)
    assert isinstance(dataset.trials_ses2, _basestring)


def test_fiac(request_mocker, tmp_path):
    # Create dummy 'files'
    fiac_dir = str(tmp_path / 'fiac_nistats' / 'nipy-data-0.2' /
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

    dataset = datasets.fetch_fiac_first_level(data_dir=str(tmp_path))
    assert isinstance(dataset.func1, _basestring)
    assert isinstance(dataset.func2, _basestring)
    assert isinstance(dataset.design_matrix1, _basestring)
    assert isinstance(dataset.design_matrix2, _basestring)
    assert isinstance(dataset.mask, _basestring)
