"""
Utilities to download NeuroImaging datasets
Author: Gael Varoquaux
"""

import fnmatch
import glob
import json
import os
import re
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets.utils import (_fetch_file,
                                    _fetch_files,
                                    _get_dataset_dir,
                                    _uncompress_file,
                                    )
from scipy.io import loadmat
from scipy.io.matlab.miobase import MatReadError
from sklearn.datasets.base import Bunch

from nistats.utils import get_data

SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in range(4, 100)]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")


def fetch_language_localizer_demo_dataset(data_dir=None, verbose=1):
    """Download language localizer demo dataset.

    Parameters
    ----------
    data_dir: string, optional
        Path to store the downloaded dataset. if None employ nilearn
        datasets default download directory.

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data_dir: string
        Path to downloaded dataset

    downloaded_files: list of string
        Absolute paths of downloaded files on disk
    """
    url = 'https://osf.io/nh987/download'
    main_folder = 'fMRI-language-localizer-demo-dataset'

    data_dir = _get_dataset_dir(main_folder, data_dir=data_dir,
                                verbose=verbose)
    # The files_spec needed for _fetch_files
    files_spec = [(main_folder + '.zip', url, {'move': main_folder + '.zip'})]
    # Only download if directory is empty
    # Directory will have been created by the call to _get_dataset_dir above
    if not os.listdir(data_dir):
        downloaded_files = _fetch_files(data_dir, files_spec, resume=True,
                                        verbose=verbose)
        _uncompress_file(downloaded_files[0])

    file_list = [os.path.join(path, f) for
                 path, dirs, files in os.walk(data_dir) for f in files]
    return data_dir, sorted(file_list)

# should be deprecated, even deleted, when the examples are adapted to use fetch_langloc_dataset
def fetch_bids_langloc_dataset(data_dir=None, verbose=1):
    """Download language localizer example bids dataset.

    Parameters
    ----------
    data_dir: string, optional
        Path to store the downloaded dataset. if None employ nilearn
        datasets default download directory.

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data_dir: string
        Path to downloaded dataset

    downloaded_files: list of string
        Absolute paths of downloaded files on disk
    """
    url = 'https://files.osf.io/v1/resources/9q7dv/providers/osfstorage/5888d9a76c613b01fc6acc4e'
    dataset_name = 'bids_langloc_example'
    main_folder = 'bids_langloc_dataset'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    # The files_spec needed for _fetch_files
    files_spec = [(main_folder + '.zip', url, {'move': main_folder + '.zip'})]
    if not os.path.exists(os.path.join(data_dir, main_folder)):
        downloaded_files = _fetch_files(data_dir, files_spec, resume=True,
                                        verbose=verbose)
        _uncompress_file(downloaded_files[0])
    main_path = os.path.join(data_dir, main_folder)
    file_list = [os.path.join(path, f) for
                 path, dirs, files in os.walk(main_path) for f in files]
    return os.path.join(data_dir, main_folder), sorted(file_list)


def fetch_openneuro_dataset_index(data_dir=None,
                                  dataset_version='ds000030_R1.0.4',
                                  verbose=1):
    """ Download a file with OpenNeuro BIDS dataset index.

    Downloading the index allows to explore the dataset directories
    to select specific files to download. The index is a sorted list of urls.

    Parameters
    ----------
    data_dir: string, optional
        Path to store the downloaded dataset. if None employ nilearn
        datasets default download directory.

    dataset_version: string, optional
        dataset version name. Assumes it is of the form [name]_[version].

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    urls_path: string
        Path to downloaded dataset index

    urls: list of string
        Sorted list of dataset directories
    """
    data_prefix = '{}/{}/uncompressed'.format(dataset_version.split('_')[0],
                                              dataset_version,
                                              )
    data_dir = _get_dataset_dir(data_prefix, data_dir=data_dir,
                                verbose=verbose)

    file_url = 'https://osf.io/86xj7/download'
    final_download_path = os.path.join(data_dir, 'urls.json')
    downloaded_file_path = _fetch_files(data_dir=data_dir,
                                        files=[(final_download_path,
                                                file_url,
                                                {'move': final_download_path}
                                                )],
                                        resume=True
                                        )
    urls_path = downloaded_file_path[0]
    with open(urls_path, 'r') as json_file:
        urls = json.load(json_file)
    return urls_path, urls


def select_from_index(urls, inclusion_filters=[], exclusion_filters=[],
                      n_subjects=None):
    """Select subset of urls with given filters.

    Parameters
    ----------
    urls: list of str
        List of dataset urls obtained from index download

    inclusion_filters: list of str, optional
        List of unix shell-style wildcard strings
        that will be used to filter the url list.
        If a filter matches the url it is retained for download.
        Multiple filters work on top of each other.
        Like an "and" logical operator, creating a more restrictive query.
        Inclusion and exclusion filters apply together.
        For example the filter '*task-rest*'' would keep only urls
        that contain the 'task-rest' string.

    exclusion_filters: list of str, optional
        List of unix shell-style wildcard strings
        that will be used to filter the url list.
        If a filter matches the url it is discarded for download.
        Multiple filters work on top of each other.
        Like an "and" logical operator, creating a more restrictive query.
        Inclusion and exclusion filters apply together.
        For example the filter '*task-rest*' would discard all urls
        that contain the 'task-rest' string.

    n_subjects: int, optional
        number of subjects to download from the dataset. All by default.

    Returns
    -------
    urls: list of string
        Sorted list of filtered dataset directories
    """
    # We apply filters to the urls
    for exclusion in exclusion_filters:
        urls = [url for url in urls if not fnmatch.fnmatch(url, exclusion)]
    for inclusion in inclusion_filters:
        urls = [url for url in urls if fnmatch.fnmatch(url, inclusion)]

    # subject selection filter
    # from the url list we infer all available subjects like 'sub-xxx/'
    subject_regex = 'sub-[a-z|A-Z|0-9]*[_./]'

    def infer_subjects(urls):
        subjects = set()
        for url in urls:
            if 'sub-' in url:
                subjects.add(re.search(subject_regex, url).group(0)[:-1])
        return sorted(subjects)

    # We get a list of subjects (for the moment the first n subjects)
    selected_subjects = set(infer_subjects(urls)[:n_subjects])
    # We exclude urls of subjects not selected
    urls = [url for url in urls if 'sub-' not in url or
            re.search(subject_regex, url).group(0)[:-1] in selected_subjects]

    return urls


def patch_openneuro_dataset(file_list):
    """Add symlinks for files not named according to latest BIDS conventions."""
    rep = {'_T1w_brainmask': '_desc-brain_mask',
           '_T1w_preproc': '_desc-preproc_T1w',
           '_T1w_space-MNI152NLin2009cAsym_brainmask': '_space-MNI152NLin2009cAsym_desc-brain_mask',
           '_T1w_space-MNI152NLin2009cAsym_class-': '_space-MNI152NLin2009cAsym_label-',
           '_T1w_space-MNI152NLin2009cAsym_preproc': '_space-MNI152NLin2009cAsym_desc-preproc_T1w',
           '_bold_confounds': '_desc-confounds_regressors',
           '_bold_space-MNI152NLin2009cAsym_brainmask':'_space-MNI152NLin2009cAsym_desc-brain_mask',
           '_bold_space-MNI152NLin2009cAsym_preproc':'_space-MNI152NLin2009cAsym_desc-preproc_bold'}
    # Create a symlink if a file with the modified filename does not exist
    for old in rep:
        for name in file_list:
            if old in name:
                if not os.path.exists(name.replace(old, rep[old])):
                    os.symlink(name, name.replace(old, rep[old]))
                name = name.replace(old, rep[old])


def fetch_openneuro_dataset(
        urls=None, data_dir=None, dataset_version='ds000030_R1.0.4',
        verbose=1):
    """Download OpenNeuro BIDS dataset.

    Parameters
    ----------
    urls: list of string, optional
        Openneuro url list of dataset files to download. If not specified
        all files of the specified dataset will be downloaded.

    data_dir: string, optional
        Path to store the downloaded dataset. if None employ nilearn
        datasets default download directory.

    dataset_version: string, optional
        dataset version name. Assumes it is of the form [name]_[version].

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data_dir: string
        Path to downloaded dataset

    downloaded_files: list of string
        Absolute paths of downloaded files on disk
    """
    data_prefix = '{}/{}/uncompressed'.format(
        dataset_version.split('_')[0], dataset_version)
    data_dir = _get_dataset_dir(data_prefix, data_dir=data_dir,
                                verbose=verbose)

    # if urls are not specified we download the complete dataset index
    if urls is None:
        _, urls = fetch_openneuro_dataset_index(
            data_dir=data_dir, dataset_version=dataset_version, verbose=verbose)

    # The files_spec needed for _fetch_files
    files_spec = []
    files_dir = []
    for url in urls:
        url_path = url.split(data_prefix + '/')[1]
        file_dir = os.path.join(data_dir, url_path)
        files_spec.append((os.path.basename(file_dir), url, {}))
        files_dir.append(os.path.dirname(file_dir))

    # download the files
    downloaded = []
    for file_spec, file_dir in zip(files_spec, files_dir):
        # Timeout errors are common in the s3 connection so we try to avoid
        # failure of the dataset download for a transient instability
        success = False
        download_attempts = 4
        while download_attempts > 0 and not success:
            try:
                downloaded_files = _fetch_files(
                    file_dir, [file_spec], resume=True, verbose=verbose)
                downloaded += downloaded_files
                success = True
            except Exception:
                download_attempts -= 1
        if not success:
            raise Exception('multiple failures downloading %s' % file_spec[1])
    patch_openneuro_dataset(downloaded)
    
    return data_dir, sorted(downloaded)


def fetch_localizer_first_level(data_dir=None, verbose=1):
    """ Download a first-level localizer fMRI dataset

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, with the keys:
        epi_img: the input 4D image
        events: a csv file describing the paardigm
    """
    url = 'https://osf.io/2bqxn/download'
    epi_img = 'sub-12069_task-localizer_space-MNI305.nii.gz'
    events = 'sub-12069_task-localizer_events.tsv'
    opts = {'uncompress': True}
    options = ('epi_img', 'events')
    dir_ = 'localizer_first_level'
    filenames = [(os.path.join(dir_, name), url, opts)
                  for name in [epi_img, events]]

    dataset_name = 'localizer_first_level'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, filenames, verbose=verbose)

    params = dict(list(zip(options, files)))
    return Bunch(**params)


def _download_spm_auditory_data(data_dir, subject_dir, subject_id):
    print('Data absent, downloading...')
    url = ('http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/'
           'MoAEpilot.zip')
    archive_path = os.path.join(subject_dir, os.path.basename(url))
    _fetch_file(url, subject_dir)
    try:
        _uncompress_file(archive_path)
    except:
        print('Archive corrupted, trying to download it again.')
        return fetch_spm_auditory(data_dir=data_dir, data_name='',
                                  subject_id=subject_id)


def _prepare_downloaded_spm_auditory_data(subject_dir):
    """ Uncompresses downloaded spm_auditory dataset and organizes
    the data into apprpriate directories.

    Parameters
    ----------
    subject_dir: string
        Path to subject's data directory.

    Returns
    -------
    _subject_data: skl.Bunch object
        Scikit-Learn Bunch object containing data of a single subject
         from the SPM Auditory dataset.

    """
    subject_data = {}
    for file_name in SPM_AUDITORY_DATA_FILES:
        file_path = os.path.join(subject_dir, file_name)
        if os.path.exists(file_path):
            subject_data[file_name] = file_path
        else:
            print('%s missing from filelist!' % file_name)
            return None

    _subject_data = {}
    _subject_data['func'] = sorted(
            [subject_data[x] for x in subject_data.keys()
             if re.match('^fM00223_0\d\d\.img$', os.path.basename(x))])

    # volumes for this dataset of shape (64, 64, 64, 1); let's fix this
    for x in _subject_data['func']:
        vol = nib.load(x)
        if len(vol.shape) == 4:
            vol = nib.Nifti1Image(get_data(vol)[:, :, :, 0],
                                  vol.affine)
            nib.save(vol, x)

    _subject_data['anat'] = [subject_data[x] for x in subject_data.keys()
                             if re.match('^sM00223_002\.img$',
                                         os.path.basename(x))][0]

    # ... same thing for anat
    vol = nib.load(_subject_data['anat'])
    if len(vol.shape) == 4:
        vol = nib.Nifti1Image(get_data(vol)[:, :, :, 0],
                              vol.affine)
        nib.save(vol, _subject_data['anat'])

    return Bunch(**_subject_data)


def _make_path_events_file_spm_auditory_data(spm_auditory_data):
    """
    Accepts data for spm_auditory dataset as Bunch
    and constructs the filepath for its events descriptor file.
    Parameters
    ----------
    spm_auditory_data: Bunch

    Returns
    -------
    events_filepath: string
        Full path to the events.tsv file for spm_auditory dataset.
    """
    events_file_location = os.path.dirname(spm_auditory_data['func'][0])
    events_filename = os.path.basename(events_file_location) + '_events.tsv'
    events_filepath = os.path.join(events_file_location, events_filename)
    return events_filepath


def _make_events_file_spm_auditory_data(events_filepath):
    """
    Accepts destination filepath including filename and
    creates the events.tsv file for the spm_auditory dataset.

    Parameters
    ----------
    events_filepath: string
        The path where the events file will be created;

    Returns
    -------
    None

    """
    tr = 7.
    epoch_duration = 6 * tr  # duration in seconds
    conditions = ['rest', 'active'] * 8
    n_blocks = len(conditions)
    duration = epoch_duration * np.ones(n_blocks)
    onset = np.linspace(0, (n_blocks - 1) * epoch_duration, n_blocks)
    events = pd.DataFrame(
            {'onset': onset, 'duration': duration, 'trial_type': conditions})
    events.to_csv(events_filepath, sep='\t', index=False,
                       columns=['onset', 'duration', 'trial_type'])


def fetch_spm_auditory(data_dir=None, data_name='spm_auditory',
                       subject_id='sub001', verbose=1):
    """Function to fetch SPM auditory single-subject data.

    Parameters
    ----------
    data_dir: string
        Path of the data directory. Used to force data storage in a specified
        location. If the data is already present there, then will simply
        glob it.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func': string list. Paths to functional images
        - 'anat': string list. Path to anat image

    References
    ----------
    :download:
        http://www.fil.ion.ucl.ac.uk/spm/data/auditory/

    """
    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)
    if not os.path.exists(subject_dir):
        _download_spm_auditory_data(data_dir, subject_dir, subject_id)
    spm_auditory_data = _prepare_downloaded_spm_auditory_data(subject_dir)
    try:
        spm_auditory_data['events']
    except KeyError:
        events_filepath = _make_path_events_file_spm_auditory_data(
                                                            spm_auditory_data)
        if not os.path.isfile(events_filepath):
            _make_events_file_spm_auditory_data(events_filepath)
        spm_auditory_data['events'] = events_filepath
    return spm_auditory_data


def _get_func_data_spm_multimodal(subject_dir, session, _subject_data):
    session_func = sorted(glob.glob(
            os.path.join(
                    subject_dir,
                    ('fMRI/Session%i/fMETHODS-000%i-*-01.img' % (
                        session, session + 4)
                     )
                    )
            ))
    if len(session_func) < 390:
        print('Missing %i functional scans for session %i.' % (
            390 - len(session_func), session))
        return None

    _subject_data['func%i' % (session)] = session_func
    return _subject_data


def _get_session_trials_spm_multimodal(subject_dir, session, _subject_data):
    sess_trials = os.path.join(
            subject_dir,
            'fMRI/trials_ses%i.mat' % (session))
    if not os.path.isfile(sess_trials):
        print('Missing session file: %s' % sess_trials)
        return None

    _subject_data['trials_ses%i' % (session)] = sess_trials
    return _subject_data


def _get_anatomical_data_spm_multimodal(subject_dir, _subject_data):
    anat = os.path.join(subject_dir, 'sMRI/smri.img')
    if not os.path.isfile(anat):
        print('Missing structural image.')
        return None

    _subject_data['anat'] = anat
    return _subject_data


def _glob_spm_multimodal_fmri_data(subject_dir):
    """glob data from subject_dir."""
    _subject_data = {'slice_order': 'descending'}

    for session in range(1, 3):
        # glob func data for session
        _subject_data = _get_func_data_spm_multimodal(subject_dir, session, _subject_data)
        if not _subject_data:
            return None
        # glob trials .mat file
        _subject_data = _get_session_trials_spm_multimodal(subject_dir, session, _subject_data)
        if not _subject_data:
            return None
        try:
            events = _make_events_file_spm_multimodal_fmri(_subject_data, session)
        except MatReadError as mat_err:
            warnings.warn('{}. An events.tsv file cannot be generated'.format(str(mat_err)))
        else:
            events_filepath = _make_events_filepath_spm_multimodal_fmri(_subject_data, session)
            events.to_csv(events_filepath, sep='\t', index=False)
            _subject_data['events{}'.format(session)] = events_filepath


    # glob for anat data
    _subject_data = _get_anatomical_data_spm_multimodal(subject_dir, _subject_data)
    if not _subject_data:
        return None

    return Bunch(**_subject_data)


def _download_data_spm_multimodal(data_dir, subject_dir, subject_id):
    print('Data absent, downloading...')
    urls = [
        # fmri
        ('http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/'
        'multimodal_fmri.zip'),

        # structural
        ('http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/'
         'multimodal_smri.zip')
        ]

    for url in urls:
        archive_path = os.path.join(subject_dir, os.path.basename(url))
        _fetch_file(url, subject_dir)
        try:
            _uncompress_file(archive_path)
        except:
            print('Archive corrupted, trying to download it again.')
            return fetch_spm_multimodal_fmri(data_dir=data_dir,
                                             data_name='',
                                             subject_id=subject_id)

    return _glob_spm_multimodal_fmri_data(subject_dir)


def _make_events_filepath_spm_multimodal_fmri(_subject_data, session):
    key = 'trials_ses{}'.format(session)
    events_file_location = os.path.dirname(_subject_data[key])
    events_filename = 'session{}_events.tsv'.format(session)
    events_filepath = os.path.join(events_file_location, events_filename)
    return events_filepath


def _make_events_file_spm_multimodal_fmri(_subject_data, session):
    tr = 2.
    timing = loadmat(_subject_data['trials_ses%i' % (session)],
                     squeeze_me=True, struct_as_record=False)
    faces_onsets = timing['onsets'][0].ravel()
    scrambled_onsets = timing['onsets'][1].ravel()
    onsets = np.hstack((faces_onsets, scrambled_onsets))
    onsets *= tr  # because onsets were reporting in 'scans' units
    conditions = (['faces'] * len(faces_onsets) +
                  ['scrambled'] * len(scrambled_onsets))
    duration = np.ones_like(onsets)
    events = pd.DataFrame({'trial_type': conditions, 'onset': onsets,
                             'duration': duration})
    return events


def fetch_spm_multimodal_fmri(data_dir=None, data_name='spm_multimodal_fmri',
                              subject_id='sub001', verbose=1):
    """Fetcher for Multi-modal Face Dataset.

    Parameters
    ----------
    data_dir: string
        path of the data directory. Used to force data storage in a specified
        location. If the data is already present there, then will simply
        glob it.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func1': string list. Paths to functional images for session 1
        - 'func2': string list. Paths to functional images for session 2
        - 'trials_ses1': string list. Path to onsets file for session 1
        - 'trials_ses2': string list. Path to onsets file for session 2
        - 'anat': string. Path to anat file

    References
    ----------
    :download:
        http://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/

    """

    data_dir = _get_dataset_dir(data_name, data_dir=data_dir, verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)

    # maybe data_dir already contains the data ?
    data = _glob_spm_multimodal_fmri_data(subject_dir)
    if data is not None:
        return data

    # No. Download the data
    return _download_data_spm_multimodal(data_dir, subject_dir, subject_id)


def fetch_fiac_first_level(data_dir=None, verbose=1):
    """ Download a first-level fiac fMRI dataset (2 sessions)

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.
    """
    data_dir = _get_dataset_dir('fiac_nistats', data_dir=data_dir,
                                verbose=verbose)
    def _glob_fiac_data():
        """glob data from subject_dir."""
        _subject_data = {}
        subject_dir = os.path.join(data_dir, 'nipy-data-0.2/data/fiac/fiac0')
        for session in [1, 2]:
            # glob func data for session
            session_func = os.path.join(subject_dir, 'run%i.nii.gz' % session)
            if not os.path.isfile(session_func):
                print('Missing functional scan for session %i.' % session)
                return None

            _subject_data['func%i' % session] = session_func

            # glob design matrix .npz file
            sess_dmtx = os.path.join(subject_dir, 'run%i_design.npz' % session)
            if not os.path.isfile(sess_dmtx):
                print('Missing session file: %s' % sess_dmtx)
                return None

            _subject_data['design_matrix%i' % session] = sess_dmtx

        # glob for mask data
        mask = os.path.join(subject_dir, 'mask.nii.gz')
        if not os.path.isfile(mask):
            print('Missing mask image.')
            return None

        _subject_data['mask'] = mask
        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_fiac_data()
    if data is not None:
        return data

    # No. Download the data
    print('Data absent, downloading...')
    url = 'http://nipy.sourceforge.net/data-packages/nipy-data-0.2.tar.gz'

    archive_path = os.path.join(data_dir, os.path.basename(url))
    _fetch_file(url, data_dir)
    try:
        _uncompress_file(archive_path)
    except:
        print('Archive corrupted, trying to download it again.')
        return fetch_fiac_first_level(data_dir=data_dir)

    return _glob_fiac_data()
