"""
Utilities to download NeuroImaging datasets
"""
# Author: Gael Varoquaux
# License: simplified BSD

import os
import re
import glob
import nibabel
from sklearn.datasets.base import Bunch

from nilearn.datasets import (
    _get_dataset_dir, _fetch_files, _fetch_file, _uncompress_file)

SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in range(4, 100)]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")


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
        path of the data directory. Used to force data storage in a specified
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
    if not data is None:
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
    if not data is None:
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
    data_dir = _get_dataset_dir('', data_dir=data_dir, verbose=verbose)

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
    if not data is None:
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
