"""
Utilities to download NeuroImaging datasets
Author: Gael Varoquaux
"""

import os
import re
import glob
import json
import nibabel
from sklearn.datasets.base import Bunch

from nilearn.datasets.utils import (
    _get_dataset_dir, _fetch_files, _fetch_file, _uncompress_file)

SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in range(4, 100)]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")


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


def fetch_openfmri_dataset(dataset_name='ds000001', dataset_revision=None,
                           data_dir=None, verbose=1):
    """Download latest revision of specified bids dataset.

    Compressed files will not be uncompressed automatically due to the expected
    great size of downloaded dataset.

    Only datasets that contain preprocessed files following the official
    conventions of the future BIDS derivatives specification can be used out
    of the box with Nistats. Otherwise custom preprocessing would need to be
    performed, optionally following the BIDS derivatives specification for the
    preprocessing output files.

    Parameters
    ----------
    dataset_name: string, optional
        Accesion number as published in https://openfmri.org/dataset/.
        Downloads by default dataset ds000001.

    dataset_revision: string, optional
        Revision as presented in the specific dataset link accesible
        from https://openfmri.org/dataset/. Looks for the latest by default.

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
    # We download a json file with all the api data from the openfmri server
    openfmri_api = 'https://openfmri.org/dataset/api'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_file(openfmri_api, data_dir)
    json_api = json.load(open(files, 'r'))

    dataset_url_set = []
    for i in range(len(json_api)):
        # We look for the desired dataset in the json api file
        if dataset_name == json_api[i]['accession_number']:
            # Now we look for the desired revision or the last one
            if not dataset_revision:
                revision = json_api[i]['revision_set']
                if revision:
                    dataset_revision = revision[-1]['revision_number']
            # After selecting the revision we download all its files
            link_set = json_api[i]['link_set']
            for link in link_set:
                revision = link['revision']
                if revision == dataset_revision:
                    dataset_url_set.append(link['url'])
            # If revision is specified but no file is found there is an issue
            if dataset_revision and not dataset_url_set:
                Exception('No files found for revision %s' % dataset_revision)
            break

    if not dataset_url_set:
        raise ValueError('dataset %s not found' % dataset_name)
    else:
        # The files_spec needed for _fetch_files
        files_spec = []
        for dat_url in dataset_url_set:
            target_file = os.path.basename(dat_url)
            url = dat_url
            files_spec.append((target_file, url, {}))
        # download the files
        downloaded_files = _fetch_files(data_dir, files_spec, resume=True,
                                        verbose=verbose)
    return data_dir, downloaded_files


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
            vol = nibabel.load(x)
            if len(vol.shape) == 4:
                vol = nibabel.Nifti1Image(vol.get_data()[:, :, :, 0],
                                          vol.get_affine())
                nibabel.save(vol, x)

        _subject_data["anat"] = [subject_data[x] for x in subject_data.keys()
                                 if re.match("^sM00223_002\.img$",
                                             os.path.basename(x))][0]

        # ... same thing for anat
        vol = nibabel.load(_subject_data["anat"])
        if len(vol.shape) == 4:
            vol = nibabel.Nifti1Image(vol.get_data()[:, :, :, 0],
                                      vol.get_affine())
            nibabel.save(vol, _subject_data["anat"])

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
