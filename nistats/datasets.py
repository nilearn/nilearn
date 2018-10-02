"""
Utilities to download NeuroImaging datasets
Author: Gael Varoquaux
"""

import fnmatch
import glob
import json
import os
import re

from botocore.handlers import disable_signing
import nibabel as nib
import pandas as pd
from nilearn.datasets.utils import (_fetch_file,
                                    _fetch_files,
                                    _get_dataset_dir,
                                    _uncompress_file,
                                    )
from sklearn.datasets.base import Bunch


SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in range(4, 100)]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")


def _check_import_boto3(module_name):
    """Helper function which checks boto3 is installed or not

    If not installed raises an ImportError with user friendly
    information.
    """
    try:
        module = __import__(module_name)
    except ImportError:
        info = "Please install boto3 to download openneuro datasets."
        raise ImportError("Module {0} cannot be found. {1} "
                          .format(module_name, info))
    return module


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


def fetch_openneuro_dataset_index(
        data_dir=None, dataset_version='ds000030_R1.0.4', verbose=1):
    """Download openneuro bids dataset index

    Downloading the index allows to explore the dataset directories
    to select specific files to download. The index is a sorted list of urls.

    Note: This function requires boto3 to be installed.

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
    boto3 = _check_import_boto3("boto3")
    data_prefix = '{}/{}/uncompressed'.format(
        dataset_version.split('_')[0], dataset_version)
    data_dir = _get_dataset_dir(data_prefix, data_dir=data_dir,
                                verbose=verbose)

    # First we download the url list from the uncompressed dataset version
    urls_path = os.path.join(data_dir, 'urls.json')
    urls = []
    if not os.path.exists(urls_path):

        def get_url(endpoint_url, bucket_name, file_key):
            return '{}/{}/{}'.format(endpoint_url, bucket_name, file_key)

        resource = boto3.resource('s3')
        resource.meta.client.meta.events.register('choose-signer.s3.*',
                                                  disable_signing)
        bucket = resource.Bucket('openneuro')

        for obj in bucket.objects.filter(Prefix=data_prefix):
            # get url of files (keys of directories end with '/')
            if obj.key[-1] != '/':
                urls.append(
                    get_url(bucket.meta.client.meta.endpoint_url,
                            bucket.name, obj.key))
        urls = sorted(urls)

        with open(urls_path, 'w') as json_file:
            json.dump(urls, json_file)
    else:
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


def fetch_openneuro_dataset(
        urls=None, data_dir=None, dataset_version='ds000030_R1.0.4',
        verbose=1):
    """Download openneuro bids dataset.

    Note: This function requires boto3 to be installed.

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
    boto3 = _check_import_boto3("boto3")
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

    return data_dir, sorted(downloaded)


def _check_bids_compliance_localizer_first_level_paradigm_file(paradigm_file):
    paradigm = pd.read_csv(paradigm_file, sep='\t')
    return list(paradigm.columns) == ['trial_type', 'onset']


def _make_localizer_first_level_paradigm_file_bids_compliant(paradigm_file):
    """ Makes the first-level localizer fMRI dataset events file
    BIDS compliant. Overwrites the original file.
        Adds headers in first row.
        Removes first column (spurious data).
        Uses Tab character as value separator.
    
    Parameters
    ----------
    paradigm_file: string
        path to the localizer_first_level dataset's events file.
    
    Returns
    -------
    None
    """
    paradigm = pd.read_csv(paradigm_file, sep=' ', header=None, index_col=None,
                           names=['session', 'trial_type', 'onset'],
                           )
    paradigm.drop(labels='session', axis=1, inplace=True)
    paradigm.to_csv(paradigm_file, sep='\t', index=False)


def fetch_localizer_first_level(data_dir=None, verbose=1):
    """ Download a first-level localizer fMRI dataset

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:
        epi_img: the input 4D image
        paradigm: a csv file describing the paardigm
    """
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'

    dataset_name = "localizer_first_level"
    files = dict(epi_img="s12069_swaloc1_corr.nii.gz",
                 paradigm="localizer_paradigm.csv")
    # The options needed for _fetch_files
    options = [(filename, os.path.join(url, filename), {})
               for _, filename in sorted(files.items())]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, options, resume=True,
                             verbose=verbose)

    params = dict(zip(sorted(files.keys()), sub_files))
    bids_compliant_paradigm = (
            _check_bids_compliance_localizer_first_level_paradigm_file(
                                            paradigm_file=params['paradigm']
                                            )
                    )
    if not bids_compliant_paradigm:
        _make_localizer_first_level_paradigm_file_bids_compliant(paradigm_file=
                                                             params['paradigm']
                                                             )
    
    return Bunch(**params)


def fetch_spm_auditory(data_dir=None, data_name='spm_auditory',
                       subject_id="sub001", verbose=1):
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

    def _glob_spm_auditory_data():
        """glob data from subject_dir.

        """

        if not os.path.exists(subject_dir):
            return None

        subject_data = {}
        for file_name in SPM_AUDITORY_DATA_FILES:
            file_path = os.path.join(subject_dir, file_name)
            if os.path.exists(file_path):
                subject_data[file_name] = file_path
            else:
                print("%s missing from filelist!" % file_name)
                return None

        _subject_data = {}
        _subject_data["func"] = sorted(
            [subject_data[x] for x in subject_data.keys()
             if re.match("^fM00223_0\d\d\.img$", os.path.basename(x))])

        # volumes for this dataset of shape (64, 64, 64, 1); let's fix this
        for x in _subject_data["func"]:
            vol = nib.load(x)
            if len(vol.shape) == 4:
                vol = nib.Nifti1Image(vol.get_data()[:, :, :, 0],
                                      vol.affine)
                nib.save(vol, x)

        _subject_data["anat"] = [subject_data[x] for x in subject_data.keys()
                                 if re.match("^sM00223_002\.img$",
                                             os.path.basename(x))][0]

        # ... same thing for anat
        vol = nib.load(_subject_data["anat"])
        if len(vol.shape) == 4:
            vol = nib.Nifti1Image(vol.get_data()[:, :, :, 0],
                                  vol.affine)
            nib.save(vol, _subject_data["anat"])

        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_spm_auditory_data()
    if data is not None:
        return data

    # No. Download the data
    print("Data absent, downloading...")
    url = ("http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/"
           "MoAEpilot.zip")
    archive_path = os.path.join(subject_dir, os.path.basename(url))
    _fetch_file(url, subject_dir)
    try:
        _uncompress_file(archive_path)
    except:
        print("Archive corrupted, trying to download it again.")
        return fetch_spm_auditory(data_dir=data_dir, data_name="",
                                  subject_id=subject_id)

    return _glob_spm_auditory_data()


def fetch_spm_multimodal_fmri(data_dir=None, data_name="spm_multimodal_fmri",
                              subject_id="sub001", verbose=1):
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

    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)

    def _glob_spm_multimodal_fmri_data():
        """glob data from subject_dir."""
        _subject_data = {'slice_order': 'descending'}

        for session in range(2):
            # glob func data for session s + 1
            session_func = sorted(glob.glob(
                    os.path.join(
                        subject_dir,
                        ("fMRI/Session%i/fMETHODS-000%i-*-01.img" % (
                                session + 1, session + 5)))))
            if len(session_func) < 390:
                print("Missing %i functional scans for session %i." % (
                    390 - len(session_func), session))
                return None

            _subject_data['func%i' % (session + 1)] = session_func

            # glob trials .mat file
            sess_trials = os.path.join(
                subject_dir,
                "fMRI/trials_ses%i.mat" % (session + 1))
            if not os.path.isfile(sess_trials):
                print("Missing session file: %s" % sess_trials)
                return None

            _subject_data['trials_ses%i' % (session + 1)] = sess_trials

        # glob for anat data
        anat = os.path.join(subject_dir, "sMRI/smri.img")
        if not os.path.isfile(anat):
            print("Missing structural image.")
            return None

        _subject_data["anat"] = anat

        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_spm_multimodal_fmri_data()
    if data is not None:
        return data

    # No. Download the data
    print("Data absent, downloading...")
    urls = [
        # fmri
        ("http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/"
        "multimodal_fmri.zip"),

        # structural
        ("http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/"
         "multimodal_smri.zip")
        ]

    for url in urls:
        archive_path = os.path.join(subject_dir, os.path.basename(url))
        _fetch_file(url, subject_dir)
        try:
            _uncompress_file(archive_path)
        except:
            print("Archive corrupted, trying to download it again.")
            return fetch_spm_multimodal_fmri(data_dir=data_dir,
                                             data_name="",
                                             subject_id=subject_id)

    return _glob_spm_multimodal_fmri_data()


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
            # glob func data for session session + 1
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
