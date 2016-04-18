"""
Downloading NeuroImaging datasets: functional datasets (task + resting-state)
"""

import collections
import glob
import json
import io
import os
import re
import sys
import warnings
from itertools import chain

import numpy as np
import nibabel
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction import DictVectorizer

from .utils import (_get_dataset_dir, _fetch_files, _get_dataset_descr,
                    _read_md5_sum_file, _tree, _filter_columns)
from .._utils import check_niimg
from .._utils.compat import BytesIO, _basestring, _urllib, _http
from .._utils.numpy_conversions import csv_to_array


def fetch_haxby_simple(data_dir=None, url=None, resume=True,
                       query_server=True, verbose=1):
    """Download and load a simple example haxby dataset.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, interest attributes are:
        'func': list of string.  Path to nifti file with bold data.
        'session_target': list of string. Path to text file containing session
        and target data.
        'mask': string. Path to nifti mask file.
        'session': list of string. Path to text file containing labels
        (can be used for LeaveOneLabelOut cross validation for example).

    References
    ----------
    `Haxby, J., Gobbini, M., Furey, M., Ishai, A., Schouten, J.,
    and Pietrini, P. (2001). Distributed and overlapping representations of
    faces and objects in ventral temporal cortex. Science 293, 2425-2430.`

    Notes
    -----
    PyMVPA provides a tutorial using this dataset :
    http://www.pymvpa.org/tutorial.html

    More informations about its structure :
    http://dev.pymvpa.org/datadb/haxby2001.html

    See `additional information
    <http://www.sciencemag.org/content/293/5539/2425>`_
    """
    # URL of the dataset. It is optional because a test uses it to test dataset
    # downloading
    if url is None:
        url = 'http://www.pymvpa.org/files/pymvpa_exampledata.tar.bz2'

    opts = {'uncompress': True}
    files = [
        (os.path.join('pymvpa-exampledata', 'attributes.txt'), url, opts),
        (os.path.join('pymvpa-exampledata', 'bold.nii.gz'), url, opts),
        (os.path.join('pymvpa-exampledata', 'mask.nii.gz'), url, opts),
        (os.path.join('pymvpa-exampledata', 'attributes_literal.txt'),
         url, opts),
    ]

    dataset_name = 'haxby2001_simple'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, files, resume=resume,
                         query_server=query_server, verbose=verbose)

    # There is a common file for the two versions of Haxby
    fdescr = _get_dataset_descr('haxby2001')

    # List of length 1 are used because haxby_simple is single-subject
    return Bunch(func=[files[1]], session_target=[files[0]], mask=files[2],
                 conditions_target=[files[3]], description=fdescr)


def fetch_haxby(data_dir=None, n_subjects=1, fetch_stimuli=False,
                url=None, resume=True, query_server=True, verbose=1):
    """Download and loads complete haxby dataset

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    n_subjects: int, optional
        Number of subjects, from 1 to 6.

    fetch_stimuli: boolean, optional
        Indicate if stimuli images must be downloaded. They will be presented
        as a dictionnary of categories.

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        'anat': string list. Paths to anatomic images.
        'func': string list. Paths to nifti file with bold data.
        'session_target': string list. Paths to text file containing
        session and target data.
        'mask': string. Path to fullbrain mask file.
        'mask_vt': string list. Paths to nifti ventral temporal mask file.
        'mask_face': string list. Paths to nifti ventral temporal mask file.
        'mask_house': string list. Paths to nifti ventral temporal mask file.
        'mask_face_little': string list. Paths to nifti ventral temporal
        mask file.
        'mask_house_little': string list. Paths to nifti ventral temporal
        mask file.

    References
    ----------
    `Haxby, J., Gobbini, M., Furey, M., Ishai, A., Schouten, J.,
    and Pietrini, P. (2001). Distributed and overlapping representations of
    faces and objects in ventral temporal cortex. Science 293, 2425-2430.`

    Notes
    -----
    PyMVPA provides a tutorial making use of this dataset:
    http://www.pymvpa.org/tutorial.html

    More information about its structure:
    http://dev.pymvpa.org/datadb/haxby2001.html

    See `additional information
    <http://www.sciencemag.org/content/293/5539/2425>`

    Run 8 in subject 5 does not contain any task labels.
    The anatomical image for subject 6 is unavailable.
    """

    if n_subjects > 6:
        warnings.warn('Warning: there are only 6 subjects')
        n_subjects = 6

    dataset_name = 'haxby2001'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # Get the mask
    url_mask = 'https://www.nitrc.org/frs/download.php/7868/mask.nii.gz'
    mask = _fetch_files(data_dir, [('mask.nii.gz', url_mask, {})],
                        resume=resume, query_server=query_server,
                        verbose=verbose)[0]

    # Dataset files
    if url is None:
        url = 'http://data.pymvpa.org/datasets/haxby2001/'
    md5sums = _fetch_files(data_dir, [('MD5SUMS', url + 'MD5SUMS', {})],
                           resume=resume, query_server=query_server,
                           verbose=verbose)[0]
    md5sums = _read_md5_sum_file(md5sums)

    # definition of dataset files
    sub_files = ['bold.nii.gz', 'labels.txt',
                 'mask4_vt.nii.gz', 'mask8b_face_vt.nii.gz',
                 'mask8b_house_vt.nii.gz', 'mask8_face_vt.nii.gz',
                 'mask8_house_vt.nii.gz', 'anat.nii.gz']
    n_files = len(sub_files)

    files = [
        (os.path.join('subj%d' % i, sub_file),
         url + 'subj%d-2010.01.14.tar.gz' % i,
         {'uncompress': True,
          'md5sum': md5sums.get('subj%d-2010.01.14.tar.gz' % i, None)})
        for i in range(1, n_subjects + 1)
        for sub_file in sub_files
        if not (sub_file == 'anat.nii.gz' and i == 6)  # no anat for sub. 6
    ]

    files = _fetch_files(data_dir, files, resume=resume,
                         query_server=query_server, verbose=verbose)

    if n_subjects == 6:
        files.append(None)  # None value because subject 6 has no anat

    kwargs = {}
    if fetch_stimuli:
        stimuli_files = [(os.path.join('stimuli', 'README'),
                          url + 'stimuli-2010.01.14.tar.gz',
                          {'uncompress': True})]
        readme = _fetch_files(data_dir, stimuli_files, resume=resume,
                              query_server=query_server, verbose=verbose)[0]
        kwargs['stimuli'] = _tree(os.path.dirname(readme), pattern='*.jpg',
                                  dictionary=True)

    fdescr = _get_dataset_descr(dataset_name)

    # return the data
    return Bunch(
        anat=files[7::n_files],
        func=files[0::n_files],
        session_target=files[1::n_files],
        mask_vt=files[2::n_files],
        mask_face=files[3::n_files],
        mask_house=files[4::n_files],
        mask_face_little=files[5::n_files],
        mask_house_little=files[6::n_files],
        mask=mask,
        description=fdescr,
        **kwargs)


def fetch_nyu_rest(n_subjects=None, sessions=[1], data_dir=None, resume=True,
                   query_server=True, verbose=1):
    """Download and loads the NYU resting-state test-retest dataset.

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load. If None is given, all the
        subjects are used.

    sessions: iterable of int, optional
        The sessions to load. Load only the first session by default.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        'func': string list. Paths to functional images.
        'anat_anon': string list. Paths to anatomic images.
        'anat_skull': string. Paths to skull-stripped images.
        'session': numpy array. List of ids corresponding to images sessions.

    Notes
    ------
    This dataset is composed of 3 sessions of 26 participants (11 males).
    For each session, three sets of data are available:

    - anatomical:

      * anonymized data (defaced thanks to BIRN defacer)
      * skullstripped data (using 3DSkullStrip from AFNI)

    - functional

    For each participant, 3 resting-state scans of 197 continuous EPI
    functional volumes were collected :

    - 39 slices
    - matrix = 64 x 64
    - acquisition voxel size = 3 x 3 x 3 mm

    Sessions 2 and 3 were conducted in a single scan session, 45 min
    apart, and were 5-16 months after Scan 1.

    All details about this dataset can be found here :
    http://cercor.oxfordjournals.org/content/19/10/2209.full

    References
    ----------
    :Documentation:
        http://www.nitrc.org/docman/?group_id=274

    :Download:
        http://www.nitrc.org/frs/?group_id=274

    :Paper to cite:
        `The Resting Brain: Unconstrained yet Reliable
        <http://cercor.oxfordjournals.org/content/19/10/2209>`_
        Z. Shehzad, A.M.C. Kelly, P.T. Reiss, D.G. Gee, K. Gotimer,
        L.Q. Uddin, S.H. Lee, D.S. Margulies, A.K. Roy, B.B. Biswal,
        E. Petkova, F.X. Castellanos and M.P. Milham.

    :Other references:
        * `The oscillating brain: Complex and Reliable
          <http://dx.doi.org/10.1016/j.neuroimage.2009.09.037>`_
          X-N. Zuo, A. Di Martino, C. Kelly, Z. Shehzad, D.G. Gee,
          D.F. Klein, F.X. Castellanos, B.B. Biswal, M.P. Milham

        * `Reliable intrinsic connectivity networks: Test-retest
          evaluation using ICA and dual regression approach
          <http://dx.doi.org/10.1016/j.neuroimage.2009.10.080>`_,
          X-N. Zuo, C. Kelly, J.S. Adelstein, D.F. Klein,
          F.X. Castellanos, M.P. Milham

    """
    fa1 = 'http://www.nitrc.org/frs/download.php/1071/NYU_TRT_session1a.tar.gz'
    fb1 = 'http://www.nitrc.org/frs/download.php/1072/NYU_TRT_session1b.tar.gz'
    fa2 = 'http://www.nitrc.org/frs/download.php/1073/NYU_TRT_session2a.tar.gz'
    fb2 = 'http://www.nitrc.org/frs/download.php/1074/NYU_TRT_session2b.tar.gz'
    fa3 = 'http://www.nitrc.org/frs/download.php/1075/NYU_TRT_session3a.tar.gz'
    fb3 = 'http://www.nitrc.org/frs/download.php/1076/NYU_TRT_session3b.tar.gz'
    fa1_opts = {'uncompress': True,
                'move': os.path.join('session1', 'NYU_TRT_session1a.tar.gz')}
    fb1_opts = {'uncompress': True,
                'move': os.path.join('session1', 'NYU_TRT_session1b.tar.gz')}
    fa2_opts = {'uncompress': True,
                'move': os.path.join('session2', 'NYU_TRT_session2a.tar.gz')}
    fb2_opts = {'uncompress': True,
                'move': os.path.join('session2', 'NYU_TRT_session2b.tar.gz')}
    fa3_opts = {'uncompress': True,
                'move': os.path.join('session3', 'NYU_TRT_session3a.tar.gz')}
    fb3_opts = {'uncompress': True,
                'move': os.path.join('session3', 'NYU_TRT_session3b.tar.gz')}

    p_anon = os.path.join('anat', 'mprage_anonymized.nii.gz')
    p_skull = os.path.join('anat', 'mprage_skullstripped.nii.gz')
    p_func = os.path.join('func', 'lfo.nii.gz')

    subs_a = ['sub05676', 'sub08224', 'sub08889', 'sub09607', 'sub14864',
              'sub18604', 'sub22894', 'sub27641', 'sub33259', 'sub34482',
              'sub36678', 'sub38579', 'sub39529']
    subs_b = ['sub45463', 'sub47000', 'sub49401', 'sub52738', 'sub55441',
              'sub58949', 'sub60624', 'sub76987', 'sub84403', 'sub86146',
              'sub90179', 'sub94293']

    # Generate the list of files by session
    anat_anon_files = [
        [(os.path.join('session1', sub, p_anon), fa1, fa1_opts)
            for sub in subs_a]
        + [(os.path.join('session1', sub, p_anon), fb1, fb1_opts)
            for sub in subs_b],
        [(os.path.join('session2', sub, p_anon), fa2, fa2_opts)
            for sub in subs_a]
        + [(os.path.join('session2', sub, p_anon), fb2, fb2_opts)
            for sub in subs_b],
        [(os.path.join('session3', sub, p_anon), fa3, fa3_opts)
            for sub in subs_a]
        + [(os.path.join('session3', sub, p_anon), fb3, fb3_opts)
            for sub in subs_b]]

    anat_skull_files = [
        [(os.path.join('session1', sub, p_skull), fa1, fa1_opts)
            for sub in subs_a]
        + [(os.path.join('session1', sub, p_skull), fb1, fb1_opts)
            for sub in subs_b],
        [(os.path.join('session2', sub, p_skull), fa2, fa2_opts)
            for sub in subs_a]
        + [(os.path.join('session2', sub, p_skull), fb2, fb2_opts)
            for sub in subs_b],
        [(os.path.join('session3', sub, p_skull), fa3, fa3_opts)
            for sub in subs_a]
        + [(os.path.join('session3', sub, p_skull), fb3, fb3_opts)
            for sub in subs_b]]

    func_files = [
        [(os.path.join('session1', sub, p_func), fa1, fa1_opts)
            for sub in subs_a]
        + [(os.path.join('session1', sub, p_func), fb1, fb1_opts)
            for sub in subs_b],
        [(os.path.join('session2', sub, p_func), fa2, fa2_opts)
            for sub in subs_a]
        + [(os.path.join('session2', sub, p_func), fb2, fb2_opts)
            for sub in subs_b],
        [(os.path.join('session3', sub, p_func), fa3, fa3_opts)
            for sub in subs_a]
        + [(os.path.join('session3', sub, p_func), fb3, fb3_opts)
            for sub in subs_b]]

    max_subjects = len(subs_a) + len(subs_b)
    # Check arguments
    if n_subjects is None:
        n_subjects = len(subs_a) + len(subs_b)
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects

    anat_anon = []
    anat_skull = []
    func = []
    session = []
    for i in sessions:
        if not (i in [1, 2, 3]):
            raise ValueError('NYU dataset session id must be in [1, 2, 3]')
        anat_anon += anat_anon_files[i - 1][:n_subjects]
        anat_skull += anat_skull_files[i - 1][:n_subjects]
        func += func_files[i - 1][:n_subjects]
        session += [i] * n_subjects

    dataset_name = 'nyu_rest'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    anat_anon = _fetch_files(data_dir, anat_anon, resume=resume,
                             query_server=query_server, verbose=verbose)
    anat_skull = _fetch_files(data_dir, anat_skull, resume=resume,
                              query_server=query_server, verbose=verbose)
    func = _fetch_files(data_dir, func, resume=resume,
                        query_server=query_server, verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    return Bunch(anat_anon=anat_anon, anat_skull=anat_skull, func=func,
                 session=session, description=fdescr)


def fetch_adhd(n_subjects=None, data_dir=None, url=None, resume=True,
               query_server=True, verbose=1):
    """Download and load the ADHD resting-state dataset.

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load from maximum of 40 subjects.
        By default, 30 subjects will be loaded. If None is given,
        all 40 subjects will be loaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data). Default: None

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'func': Paths to functional resting-state images
         - 'phenotypic': Explanations of preprocessing steps
         - 'confounds': CSV files containing the nuisance variables

    References
    ----------
    :Download:
        ftp://www.nitrc.org/fcon_1000/htdocs/indi/adhd200/sites/ADHD200_40sub_preprocessed.tgz

    """

    if url is None:
        url = 'https://www.nitrc.org/frs/download.php/'

    # Preliminary checks and declarations
    dataset_name = 'adhd'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    ids = ['0010042', '0010064', '0010128', '0021019', '0023008', '0023012',
           '0027011', '0027018', '0027034', '0027037', '1019436', '1206380',
           '1418396', '1517058', '1552181', '1562298', '1679142', '2014113',
           '2497695', '2950754', '3007585', '3154996', '3205761', '3520880',
           '3624598', '3699991', '3884955', '3902469', '3994098', '4016887',
           '4046678', '4134561', '4164316', '4275075', '6115230', '7774305',
           '8409791', '8697774', '9744150', '9750701']
    nitrc_ids = range(7782, 7822)
    max_subjects = len(ids)
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects
    ids = ids[:n_subjects]
    nitrc_ids = nitrc_ids[:n_subjects]

    opts = dict(uncompress=True)

    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)

    # First, get the metadata
    phenotypic = ('ADHD200_40subs_motion_parameters_and_phenotypics.csv',
                  url + '7781/adhd40_metadata.tgz', opts)

    phenotypic = _fetch_files(data_dir, [phenotypic], resume=resume,
                              query_server=query_server, verbose=verbose)[0]

    # Load the csv file
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',',
                               dtype=None)

    # Keep phenotypic information for selected subjects
    int_ids = np.asarray(ids, dtype=int)
    phenotypic = phenotypic[[np.where(phenotypic['Subject'] == i)[0][0]
                             for i in int_ids]]

    # Download dataset files

    archives = [url + '%i/adhd40_%s.tgz' % (ni, ii)
                for ni, ii in zip(nitrc_ids, ids)]
    functionals = ['data/%s/%s_rest_tshift_RPI_voreg_mni.nii.gz' % (i, i)
                   for i in ids]
    confounds = ['data/%s/%s_regressors.csv' % (i, i) for i in ids]

    functionals = _fetch_files(
        data_dir, zip(functionals, archives, (opts,) * n_subjects),
        resume=resume, query_server=query_server, verbose=verbose)

    confounds = _fetch_files(
        data_dir, zip(confounds, archives, (opts,) * n_subjects),
        resume=resume, query_server=query_server, verbose=verbose)

    return Bunch(func=functionals, confounds=confounds,
                 phenotypic=phenotypic, description=fdescr)


def fetch_miyawaki2008(data_dir=None, url=None, resume=True,
                       query_server=True, verbose=1):
    """Download and loads Miyawaki et al. 2008 dataset (153MB)

    Parameters
    ----------

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :

        - 'func': string list
            Paths to nifti file with bold data
        - 'label': string list
            Paths to text file containing session and target data
        - 'mask': string
            Path to nifti mask file to define target volume in visual
            cortex
        - 'background': string
            Path to nifti file containing a background image usable as a
            background image for miyawaki images.

    References
    ----------
    `Visual image reconstruction from human brain activity
    using a combination of multiscale local image decoders
    <http://www.cell.com/neuron/abstract/S0896-6273%2808%2900958-6>`_,
    Miyawaki, Y., Uchida, H., Yamashita, O., Sato, M. A.,
    Morito, Y., Tanabe, H. C., ... & Kamitani, Y. (2008).
    Neuron, 60(5), 915-929.

    Notes
    -----
    This dataset is available on the `brainliner website
    <http://brainliner.jp/data/brainliner-admin/Reconstruct>`_

    See `additional information
    <http://www.cns.atr.jp/dni/en/downloads/
    fmri-data-set-for-visual-image-reconstruction/>`_
    """

    url = 'https://www.nitrc.org/frs/download.php' \
          '/8486/miyawaki2008.tgz?i_agree=1&download_now=1'
    opts = {'uncompress': True}

    # Dataset files

    # Functional MRI:
    #   * 20 random scans (usually used for training)
    #   * 12 figure scans (usually used for testing)

    func_figure = [(os.path.join('func', 'data_figure_run%02d.nii.gz' % i),
                    url, opts) for i in range(1, 13)]

    func_random = [(os.path.join('func', 'data_random_run%02d.nii.gz' % i),
                    url, opts) for i in range(1, 21)]

    # Labels, 10x10 patches, stimuli shown to the subject:
    #   * 20 random labels
    #   * 12 figure labels (letters and shapes)

    label_filename = 'data_%s_run%02d_label.csv'
    label_figure = [(os.path.join('label', label_filename % ('figure', i)),
                     url, opts) for i in range(1, 13)]

    label_random = [(os.path.join('label', label_filename % ('random', i)),
                     url, opts) for i in range(1, 21)]

    # Masks

    file_mask = [
        'mask.nii.gz',
        'LHlag0to1.nii.gz',
        'LHlag10to11.nii.gz',
        'LHlag1to2.nii.gz',
        'LHlag2to3.nii.gz',
        'LHlag3to4.nii.gz',
        'LHlag4to5.nii.gz',
        'LHlag5to6.nii.gz',
        'LHlag6to7.nii.gz',
        'LHlag7to8.nii.gz',
        'LHlag8to9.nii.gz',
        'LHlag9to10.nii.gz',
        'LHV1d.nii.gz',
        'LHV1v.nii.gz',
        'LHV2d.nii.gz',
        'LHV2v.nii.gz',
        'LHV3A.nii.gz',
        'LHV3.nii.gz',
        'LHV4v.nii.gz',
        'LHVP.nii.gz',
        'RHlag0to1.nii.gz',
        'RHlag10to11.nii.gz',
        'RHlag1to2.nii.gz',
        'RHlag2to3.nii.gz',
        'RHlag3to4.nii.gz',
        'RHlag4to5.nii.gz',
        'RHlag5to6.nii.gz',
        'RHlag6to7.nii.gz',
        'RHlag7to8.nii.gz',
        'RHlag8to9.nii.gz',
        'RHlag9to10.nii.gz',
        'RHV1d.nii.gz',
        'RHV1v.nii.gz',
        'RHV2d.nii.gz',
        'RHV2v.nii.gz',
        'RHV3A.nii.gz',
        'RHV3.nii.gz',
        'RHV4v.nii.gz',
        'RHVP.nii.gz'
    ]

    file_mask = [(os.path.join('mask', m), url, opts) for m in file_mask]

    file_names = func_figure + func_random + \
        label_figure + label_random + \
        file_mask

    dataset_name = 'miyawaki2008'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, file_names, resume=resume,
                         query_server=query_server, verbose=verbose)

    # Fetch the background image
    bg_img = _fetch_files(data_dir, [('bg.nii.gz', url, opts)], resume=resume,
                          verbose=verbose)[0]

    fdescr = _get_dataset_descr(dataset_name)

    # Return the data
    return Bunch(
        func=files[:32],
        label=files[32:64],
        mask=files[64],
        mask_roi=files[65:],
        background=bg_img,
        description=fdescr)


def fetch_localizer_contrasts(contrasts, n_subjects=None, get_tmaps=False,
                              get_masks=False, get_anats=False,
                              data_dir=None, url=None, resume=True,
                              query_server=True, verbose=1):
    """Download and load Brainomics Localizer dataset (94 subjects).

    "The Functional Localizer is a simple and fast acquisition
    procedure based on a 5-minute functional magnetic resonance
    imaging (fMRI) sequence that can be run as easily and as
    systematically as an anatomical scan. This protocol captures the
    cerebral bases of auditory and visual perception, motor actions,
    reading, language comprehension and mental calculation at an
    individual level. Individual functional maps are reliable and
    quite precise. The procedure is decribed in more detail on the
    Functional Localizer page."
    (see http://brainomics.cea.fr/localizer/)

    "Scientific results obtained using this dataset are described in
    Pinel et al., 2007" [1]

    Parameters
    ----------
    contrasts: list of str
        The contrasts to be fetched (for all 94 subjects available).
        Allowed values are::

            {"checkerboard",
            "horizontal checkerboard",
            "vertical checkerboard",
            "horizontal vs vertical checkerboard",
            "vertical vs horizontal checkerboard",
            "sentence listening",
            "sentence reading",
            "sentence listening and reading",
            "sentence reading vs checkerboard",
            "calculation (auditory cue)",
            "calculation (visual cue)",
            "calculation (auditory and visual cue)",
            "calculation (auditory cue) vs sentence listening",
            "calculation (visual cue) vs sentence reading",
            "calculation vs sentences",
            "calculation (auditory cue) and sentence listening",
            "calculation (visual cue) and sentence reading",
            "calculation and sentence listening/reading",
            "calculation (auditory cue) and sentence listening vs "
            "calculation (visual cue) and sentence reading",
            "calculation (visual cue) and sentence reading vs checkerboard",
            "calculation and sentence listening/reading vs button press",
            "left button press (auditory cue)",
            "left button press (visual cue)",
            "left button press",
            "left vs right button press",
            "right button press (auditory cue)",
            "right button press (visual cue)",
            "right button press",
            "right vs left button press",
            "button press (auditory cue) vs sentence listening",
            "button press (visual cue) vs sentence reading",
            "button press vs calculation and sentence listening/reading"}

        or equivalently on can use the original names::

            {"checkerboard",
            "horizontal checkerboard",
            "vertical checkerboard",
            "horizontal vs vertical checkerboard",
            "vertical vs horizontal checkerboard",
            "auditory sentences",
            "visual sentences",
            "auditory&visual sentences",
            "visual sentences vs checkerboard",
            "auditory calculation",
            "visual calculation",
            "auditory&visual calculation",
            "auditory calculation vs auditory sentences",
            "visual calculation vs sentences",
            "auditory&visual calculation vs sentences",
            "auditory processing",
            "visual processing",
            "visual processing vs auditory processing",
            "auditory processing vs visual processing",
            "visual processing vs checkerboard",
            "cognitive processing vs motor",
            "left auditory click",
            "left visual click",
            "left auditory&visual click",
            "left auditory & visual click vs right auditory&visual click",
            "right auditory click",
            "right visual click",
            "right auditory&visual click",
            "right auditory & visual click vs left auditory&visual click",
            "auditory click vs auditory sentences",
            "visual click vs visual sentences",
            "auditory&visual motor vs cognitive processing"}

    n_subjects: int, optional
        The number of subjects to load. If None is given,
        all 94 subjects are used.

    get_tmaps: boolean
        Whether t maps should be fetched or not.

    get_masks: boolean
        Whether individual masks should be fetched or not.

    get_anats: boolean
        Whether individual structural images should be fetched or not.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :

        - 'cmaps': string list
            Paths to nifti contrast maps
        - 'tmaps' string list (if 'get_tmaps' set to True)
            Paths to nifti t maps
        - 'masks': string list
            Paths to nifti files corresponding to the subjects individual masks
        - 'anats': string
            Path to nifti files corresponding to the subjects structural images

    References
    ----------
    Pinel, Philippe, et al.
    "Fast reproducible identification and large-scale databasing of
    individual functional cognitive networks."
    BMC neuroscience 8.1 (2007): 91.

    """
    if isinstance(contrasts, _basestring):
        raise ValueError('Contrasts should be a list of strings, but '
                         'a single string was given: "%s"' % contrasts)
    if n_subjects is None:
        n_subjects = 94  # 94 subjects available
    if (n_subjects > 94) or (n_subjects < 1):
        warnings.warn("Wrong value for \'n_subjects\' (%d). The maximum "
                      "value will be used instead (\'n_subjects=94\')")
        n_subjects = 94  # 94 subjects available

    # we allow the user to use alternatives to Brainomics contrast names
    contrast_name_wrapper = {
        # Checkerboard
        "checkerboard": "checkerboard",
        "horizontal checkerboard": "horizontal checkerboard",
        "vertical checkerboard": "vertical checkerboard",
        "horizontal vs vertical checkerboard":
            "horizontal vs vertical checkerboard",
        "vertical vs horizontal checkerboard":
            "vertical vs horizontal checkerboard",
        # Sentences
        "sentence listening": "auditory sentences",
        "sentence reading": "visual sentences",
        "sentence listening and reading": "auditory&visual sentences",
        "sentence reading vs checkerboard": "visual sentences vs checkerboard",
        # Calculation
        "calculation (auditory cue)": "auditory calculation",
        "calculation (visual cue)": "visual calculation",
        "calculation (auditory and visual cue)": "auditory&visual calculation",
        "calculation (auditory cue) vs sentence listening":
            "auditory calculation vs auditory sentences",
        "calculation (visual cue) vs sentence reading":
            "visual calculation vs sentences",
        "calculation vs sentences": "auditory&visual calculation vs sentences",
        # Calculation + Sentences
        "calculation (auditory cue) and sentence listening":
            "auditory processing",
        "calculation (visual cue) and sentence reading":
            "visual processing",
        "calculation (visual cue) and sentence reading vs "
        "calculation (auditory cue) and sentence listening":
            "visual processing vs auditory processing",
        "calculation (auditory cue) and sentence listening vs "
        "calculation (visual cue) and sentence reading":
            "auditory processing vs visual processing",
        "calculation (visual cue) and sentence reading vs checkerboard":
            "visual processing vs checkerboard",
        "calculation and sentence listening/reading vs button press":
            "cognitive processing vs motor",
        # Button press
        "left button press (auditory cue)": "left auditory click",
        "left button press (visual cue)": "left visual click",
        "left button press": "left auditory&visual click",
        "left vs right button press": "left auditory & visual click vs "
            + "right auditory&visual click",
        "right button press (auditory cue)": "right auditory click",
        "right button press (visual cue)": "right visual click",
        "right button press": "right auditory & visual click",
        "right vs left button press": "right auditory & visual click "
            + "vs left auditory&visual click",
        "button press (auditory cue) vs sentence listening":
            "auditory click vs auditory sentences",
        "button press (visual cue) vs sentence reading":
            "visual click vs visual sentences",
        "button press vs calculation and sentence listening/reading":
            "auditory&visual motor vs cognitive processing"}
    allowed_contrasts = list(contrast_name_wrapper.values())
    # convert contrast names
    contrasts_wrapped = []
    # get a unique ID for each contrast. It is used to give a unique name to
    # each download file and avoid name collisions.
    contrasts_indices = []
    for contrast in contrasts:
        if contrast in allowed_contrasts:
            contrasts_wrapped.append(contrast)
            contrasts_indices.append(allowed_contrasts.index(contrast))
        elif contrast in contrast_name_wrapper:
            name = contrast_name_wrapper[contrast]
            contrasts_wrapped.append(name)
            contrasts_indices.append(allowed_contrasts.index(name))
        else:
            raise ValueError("Contrast \'%s\' is not available" % contrast)

    # It is better to perform several small requests than a big one because:
    # - Brainomics server has no cache (can lead to timeout while the archive
    #   is generated on the remote server)
    # - Local (cached) version of the files can be checked for each contrast
    opts = {'uncompress': True}
    subject_ids = ["S%02d" % s for s in range(1, n_subjects + 1)]
    subject_id_max = subject_ids[-1]
    data_types = ["c map"]
    if get_tmaps:
        data_types.append("t map")
    rql_types = str.join(", ", ["\"%s\"" % x for x in data_types])
    root_url = "http://brainomics.cea.fr/localizer/"

    base_query = ("Any X,XT,XL,XI,XF,XD WHERE X is Scan, X type XT, "
                  "X concerns S, "
                  "X label XL, X identifier XI, "
                  "X format XF, X description XD, "
                  'S identifier <= "%s", ' % (subject_id_max, ) +
                  'X type IN(%(types)s), X label "%(label)s"')

    urls = ["%sbrainomics_data_%d.zip?rql=%s&vid=data-zip"
            % (root_url, i,
               _urllib.parse.quote(base_query % {"types": rql_types,
                                                 "label": c},
                                   safe=',()'))
            for c, i in zip(contrasts_wrapped, contrasts_indices)]
    filenames = []
    for subject_id in subject_ids:
        for data_type in data_types:
            for contrast_id, contrast in enumerate(contrasts_wrapped):
                name_aux = str.replace(
                    str.join('_', [data_type, contrast]), ' ', '_')
                file_path = os.path.join(
                    "brainomics_data", subject_id, "%s.nii.gz" % name_aux)
                file_tarball_url = urls[contrast_id]
                filenames.append((file_path, file_tarball_url, opts))
    # Fetch masks if asked by user
    if get_masks:
        urls.append("%sbrainomics_data_masks.zip?rql=%s&vid=data-zip"
                    % (root_url,
                       _urllib.parse.quote(base_query % {
                           "types": '"boolean mask"',
                           "label": "mask"}, safe=',()')))
        for subject_id in subject_ids:
            file_path = os.path.join(
                "brainomics_data", subject_id, "boolean_mask_mask.nii.gz")
            file_tarball_url = urls[-1]
            filenames.append((file_path, file_tarball_url, opts))
    # Fetch anats if asked by user
    if get_anats:
        urls.append("%sbrainomics_data_anats.zip?rql=%s&vid=data-zip"
                    % (root_url,
                       _urllib.parse.quote(base_query % {
                           "types": '"normalized T1"',
                           "label": "anatomy"}, safe=',()')))
        for subject_id in subject_ids:
            file_path = os.path.join(
                "brainomics_data", subject_id,
                "normalized_T1_anat_defaced.nii.gz")
            file_tarball_url = urls[-1]
            filenames.append((file_path, file_tarball_url, opts))
    # Fetch subject characteristics (separated in two files)
    if url is None:
        url_csv = ("%sdataset/cubicwebexport.csv?rql=%s&vid=csvexport"
                   % (root_url,
                      _urllib.parse.quote("Any X WHERE X is Subject")))
        url_csv2 = ("%sdataset/cubicwebexport2.csv?rql=%s&vid=csvexport"
                    % (root_url,
                       _urllib.parse.quote(
                           "Any X,XI,XD WHERE X is QuestionnaireRun, "
                           "X identifier XI, X datetime XD", safe=',')))
    else:
        url_csv = "%s/cubicwebexport.csv" % url
        url_csv2 = "%s/cubicwebexport2.csv" % url
    filenames += [("cubicwebexport.csv", url_csv, {}),
                  ("cubicwebexport2.csv", url_csv2, {})]

    # Actual data fetching
    dataset_name = 'brainomics_localizer'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    fdescr = _get_dataset_descr(dataset_name)
    files = _fetch_files(data_dir, filenames, resume=resume,
                         query_server=query_server, verbose=verbose)
    anats = None
    masks = None
    tmaps = None
    # combine data from both covariates files into one single recarray
    from numpy.lib.recfunctions import join_by
    ext_vars_file2 = files[-1]
    csv_data2 = np.recfromcsv(ext_vars_file2, delimiter=';')
    files = files[:-1]
    ext_vars_file = files[-1]
    csv_data = np.recfromcsv(ext_vars_file, delimiter=';')
    files = files[:-1]
    # join_by sorts the output along the key
    csv_data = join_by('subject_id', csv_data, csv_data2,
                       usemask=False, asrecarray=True)[:n_subjects]
    if get_anats:
        anats = files[-n_subjects:]
        files = files[:-n_subjects]
    if get_masks:
        masks = files[-n_subjects:]
        files = files[:-n_subjects]
    if get_tmaps:
        tmaps = files[1::2]
        files = files[::2]
    return Bunch(cmaps=files, tmaps=tmaps, masks=masks, anats=anats,
                 ext_vars=csv_data, description=fdescr)


def fetch_localizer_calculation_task(n_subjects=None, data_dir=None, url=None,
                                     resume=True, query_server=True,
                                     verbose=1):
    """Fetch calculation task contrast maps from the localizer.

    This function is only a caller for the fetch_localizer_contrasts in order
    to simplify examples reading and understanding.
    The 'calculation (auditory and visual cue)' contrast is used.

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load. If None is given,
        all 94 subjects are used.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'cmaps': string list, giving paths to nifti contrast maps

    """
    data = fetch_localizer_contrasts(["calculation (auditory and visual cue)"],
                                     n_subjects=n_subjects,
                                     get_tmaps=False, get_masks=False,
                                     get_anats=False, data_dir=data_dir,
                                     url=url, resume=resume, verbose=verbose,
                                     query_server=query_server)
    data.pop('tmaps')
    data.pop('masks')
    data.pop('anats')
    return data


def fetch_abide_pcp(data_dir=None, n_subjects=None, pipeline='cpac',
                    band_pass_filtering=False, global_signal_regression=False,
                    derivatives=['func_preproc'], quality_checked=True,
                    url=None, resume=True, query_server=True, verbose=1,
                    **kwargs):
    """ Fetch ABIDE dataset

    Fetch the Autism Brain Imaging Data Exchange (ABIDE) dataset wrt criteria
    that can be passed as parameter. Note that this is the preprocessed
    version of ABIDE provided by the preprocess connectome projects (PCP).

    Parameters
    ----------

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    n_subjects: int, optional
        The number of subjects to load. If None is given,
        all 94 subjects are used.

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int
        Verbosity level (0 means no message).

    pipeline: string, optional
        Possible pipelines are "ccs", "cpac", "dparsf" and "niak"

    band_pass_filtering: boolean, optional
        Due to controversies in the literature, band pass filtering is
        optional. If true, signal is band filtered between 0.01Hz and 0.1Hz.

    global_signal_regression: boolean optional
        Indicates if global signal regression should be applied on the
        signals.

    derivatives: string list, optional
        Types of downloaded files. Possible values are: alff, degree_binarize,
        degree_weighted, dual_regression, eigenvector_binarize,
        eigenvector_weighted, falff, func_mask, func_mean, func_preproc, lfcd,
        reho, rois_aal, rois_cc200, rois_cc400, rois_dosenbach160, rois_ez,
        rois_ho, rois_tt, and vmhc. Please refer to the PCP site for more
        details.

    quality_checked: boolean, optional
        if true (default), restrict the list of the subjects to the one that
        passed quality assessment for all raters.

    kwargs: parameter list, optional
        Any extra keyword argument will be used to filter downloaded subjects
        according to the CSV phenotypic file. Some examples of filters are
        indicated below.

    SUB_ID: list of integers in [50001, 50607], optional
        Ids of the subjects to be loaded.

    DX_GROUP: integer in {1, 2}, optional
        1 is autism, 2 is control

    DSM_IV_TR: integer in [0, 4], optional
        O is control, 1 is autism, 2 is Asperger, 3 is PPD-NOS,
        4 is Asperger or PPD-NOS

    AGE_AT_SCAN: float in [6.47, 64], optional
        Age of the subject

    SEX: integer in {1, 2}, optional
        1 is male, 2 is female

    HANDEDNESS_CATEGORY: string in {'R', 'L', 'Mixed', 'Ambi'}, optional
        R = Right, L = Left, Ambi = Ambidextrous

    HANDEDNESS_SCORE: integer in [-100, 100], optional
        Positive = Right, Negative = Left, 0 = Ambidextrous

    Notes
    -----
    Code and description of preprocessing pipelines are provided on the
    `PCP website <http://preprocessed-connectomes-project.github.io/>`.

    References
    ----------
    Nielsen, Jared A., et al. "Multisite functional connectivity MRI
    classification of autism: ABIDE results." Frontiers in human neuroscience
    7 (2013).
    """
    # People keep getting it wrong and submiting a string instead of a
    # list of strings. We'll make their life easy
    if isinstance(derivatives, _basestring):
        derivatives = [derivatives, ]

    # Parameter check
    for derivative in derivatives:
        if derivative not in [
                'alff', 'degree_binarize', 'degree_weighted',
                'dual_regression', 'eigenvector_binarize',
                'eigenvector_weighted', 'falff', 'func_mask', 'func_mean',
                'func_preproc', 'lfcd', 'reho', 'rois_aal', 'rois_cc200',
                'rois_cc400', 'rois_dosenbach160', 'rois_ez', 'rois_ho',
                'rois_tt', 'vmhc']:
            raise KeyError('%s is not a valid derivative' % derivative)

    strategy = ''
    if not band_pass_filtering:
        strategy += 'no'
    strategy += 'filt_'
    if not global_signal_regression:
        strategy += 'no'
    strategy += 'global'

    # General file: phenotypic information
    dataset_name = 'ABIDE_pcp'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    if url is None:
        url = ('https://s3.amazonaws.com/fcp-indi/data/Projects/'
               'ABIDE_Initiative')

    if quality_checked:
        kwargs['qc_rater_1'] = b'OK'
        kwargs['qc_anat_rater_2'] = [b'OK', b'maybe']
        kwargs['qc_func_rater_2'] = [b'OK', b'maybe']
        kwargs['qc_anat_rater_3'] = b'OK'
        kwargs['qc_func_rater_3'] = b'OK'

    # Fetch the phenotypic file and load it
    csv = 'Phenotypic_V1_0b_preprocessed1.csv'
    path_csv = _fetch_files(data_dir, [(csv, url + '/' + csv, {})],
                            resume=resume, query_server=query_server,
                            verbose=verbose)[0]

    # Note: the phenotypic file contains string that contains comma which mess
    # up numpy array csv loading. This is why I do a pass to remove the last
    # field. This can be
    # done simply with pandas but we don't want such dependency ATM
    # pheno = pandas.read_csv(path_csv).to_records()
    with open(path_csv, 'r') as pheno_f:
        pheno = ['i' + pheno_f.readline()]

        # This regexp replaces commas between double quotes
        for line in pheno_f:
            pheno.append(re.sub(r',(?=[^"]*"(?:[^"]*"[^"]*")*[^"]*$)',
                         ";", line))

    # bytes (encode()) needed for python 2/3 compat with numpy
    pheno = '\n'.join(pheno).encode()
    pheno = BytesIO(pheno)
    pheno = np.recfromcsv(pheno, comments='$', case_sensitive=True)

    # First, filter subjects with no filename
    pheno = pheno[pheno['FILE_ID'] != b'no_filename']
    # Apply user defined filters
    user_filter = _filter_columns(pheno, kwargs)
    pheno = pheno[user_filter]

    # Go into specific data folder and url
    data_dir = os.path.join(data_dir, pipeline, strategy)
    url = '/'.join([url, 'Outputs', pipeline, strategy])

    # Get the files
    results = {}
    file_ids = [file_id.decode() for file_id in pheno['FILE_ID']]
    if n_subjects is not None:
        file_ids = file_ids[:n_subjects]
        pheno = pheno[:n_subjects]

    results['description'] = _get_dataset_descr(dataset_name)
    results['phenotypic'] = pheno
    for derivative in derivatives:
        ext = '.1D' if derivative.startswith('rois') else '.nii.gz'
        files = [(fid + '_' + derivative + ext,
                  '/'.join([url, derivative, fid + '_' + derivative + ext]),
                  {}) for fid in file_ids]
        files = _fetch_files(data_dir, files, resume=resume,
                             query_server=query_server, verbose=verbose)
        # Load derivatives if needed
        if ext == '.1D':
            files = [np.loadtxt(f) for f in files]
        results[derivative] = files
    return Bunch(**results)


def _load_mixed_gambles(zmap_imgs):
    """Ravel zmaps (one per subject) along time axis, resulting,
    in a n_subjects * n_trials 3D niimgs and, and then make
    gain vector y of same length.
    """
    X = []
    y = []
    mask = []
    for zmap_img in zmap_imgs:
        # load subject data
        this_X = zmap_img.get_data()
        affine = zmap_img.get_affine()
        finite_mask = np.all(np.isfinite(this_X), axis=-1)
        this_mask = np.logical_and(np.all(this_X != 0, axis=-1),
                                   finite_mask)
        this_y = np.array([np.arange(1, 9)] * 6).ravel()

        # gain levels
        if len(this_y) != this_X.shape[-1]:
            raise RuntimeError("%s: Expecting %i volumes, got %i!" % (
                zmap_img, len(this_y), this_X.shape[-1]))

        # standardize subject data
        this_X -= this_X.mean(axis=-1)[..., np.newaxis]
        std = this_X.std(axis=-1)
        std[std == 0] = 1
        this_X /= std[..., np.newaxis]

        # commit subject data
        X.append(this_X)
        y.extend(this_y)
        mask.append(this_mask)
    y = np.array(y)
    X = np.concatenate(X, axis=-1)
    mask = np.sum(mask, axis=0) > .5 * len(mask)
    mask = np.logical_and(mask, np.all(np.isfinite(X), axis=-1))
    X = X[mask, :].T
    tmp = np.zeros(list(mask.shape) + [len(X)])
    tmp[mask, :] = X.T
    mask_img = nibabel.Nifti1Image(mask.astype(np.int), affine)
    X = nibabel.four_to_three(nibabel.Nifti1Image(tmp, affine))
    return X, y, mask_img


def fetch_mixed_gambles(n_subjects=1, data_dir=None, url=None, resume=True,
                        return_raw_data=False, query_server=True, verbose=0):
    """Fetch Jimura "mixed gambles" dataset.

    Parameters
    ----------
    n_subjects: int, optional (default 1)
        The number of subjects to load. If None is given, all the
        subjects are used.

    data_dir: string, optional (default None)
        Path of the data directory. Used to force data storage in a specified
        location. Default: None.

    url: string, optional (default None)
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume: bool
        Whether to resume download of a partly-downloaded file.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int, optional (default 0)
        Defines the level of verbosity of the output.

    return_raw_data: bool, optional (default True)
        If false, then the data will transformed into and (X, y) pair, suitable
        for machine learning routines. X is a list of n_subjects * 48
        Nifti1Image objects (where 48 is the number of trials),
        and y is an array of shape (n_subjects * 48,).

    smooth: float, or list of 3 floats, optional (default 0.)
        Size of smoothing kernel to apply to the loaded zmaps.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'zmaps': string list
            Paths to realigned gain betamaps (one nifti per subject).
        'gain': ..
            If make_Xy is true, this is a list of n_subjects * 48
            Nifti1Image objects, else it is None.
        'y': array of shape (n_subjects * 48,) or None
            If make_Xy is true, then this is an array of shape
            (n_subjects * 48,), else it is None.

    References
    ----------
    [1] K. Jimura and R. Poldrack, "Analyses of regional-average activation
        and multivoxel pattern information tell complementary stories",
        Neuropsychologia, vol. 50, page 544, 2012
    """
    if n_subjects > 16:
        warnings.warn('Warning: there are only 16 subjects!')
        n_subjects = 16
    if url is None:
        url = ("https://www.nitrc.org/frs/download.php/7229/"
               "jimura_poldrack_2012_zmaps.zip")
    opts = dict(uncompress=True)
    files = [("zmaps%ssub%03i_zmaps.nii.gz" % (os.sep, (j + 1)), url, opts)
             for j in range(n_subjects)]
    data_dir = _get_dataset_dir('jimura_poldrack_2012_zmaps',
                                data_dir=data_dir)
    zmap_fnames = _fetch_files(data_dir, files, resume=resume,
                               query_server=query_server, verbose=verbose)
    subject_id = np.repeat(np.arange(n_subjects), 6 * 8)
    data = Bunch(zmaps=zmap_fnames,
                 subject_id=subject_id)
    if not return_raw_data:
        X, y, mask_img = _load_mixed_gambles(check_niimg(data.zmaps,
                                                         return_iterator=True))
        data.zmaps, data.gain, data.mask_img = X, y, mask_img
    return data


def fetch_megatrawls_netmats(dimensionality=100, timeseries='eigen_regression',
                             matrices='partial_correlation', data_dir=None,
                             resume=True, verbose=1):
    """Downloads and returns Network Matrices data from MegaTrawls release in HCP.

    This data can be used to predict relationships between imaging data and
    non-imaging behavioural measures such as age, sex, education, etc.
    The network matrices are estimated from functional connectivity
    datasets of 461 subjects. Full technical details in [1] [2].

    .. versionadded:: 0.2.2

    Parameters
    ----------
    dimensionality: int, optional
        Valid inputs are 25, 50, 100, 200, 300. By default, network matrices
        estimated using Group ICA brain parcellations of 100 components/dimensions
        will be returned.

    timeseries: str, optional
        Valid inputs are 'multiple_spatial_regression' or 'eigen_regression'. By
        default 'eigen_regression', matrices estimated using first principal
        eigen component timeseries signals extracted from each subject data
        parcellations will be returned. Otherwise, 'multiple_spatial_regression'
        matrices estimated using spatial regressor based timeseries signals
        extracted from each subject data parcellations will be returned.

    matrices: str, optional
        Valid inputs are 'full_correlation' or 'partial_correlation'. By default,
        partial correlation matrices will be returned otherwise if selected
        full correlation matrices will be returned.

    data_dir: str, default is None, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    resume: bool, default is True
        This parameter is required if a partially downloaded file is needed
        to be resumed to download again.

    verbose: int, default is 1
        This parameter is used to set the verbosity level to print the message
        to give information about the processing.
        0 indicates no information will be given.

    Returns
    -------
    data: Bunch
        dictionary-like object, the attributes are :

        - 'dimensions': int, consists of given input in dimensions.

        - 'timeseries': str, consists of given input in timeseries method.

        - 'matrices': str, consists of given type of specific matrices.

        - 'correlation_matrices': ndarray, consists of correlation matrices
          based on given type of matrices. Array size will depend on given
          dimensions (n, n).
        - 'description': data description

    References
    ----------
    [1] Stephen Smith et al, HCP beta-release of the Functional Connectivity
        MegaTrawl.
        April 2015 "HCP500-MegaTrawl" release.
        https://db.humanconnectome.org/megatrawl/

    [2] Smith, S.M. et al. Nat. Neurosci. 18, 1565-1567 (2015).

    [3] N.Filippini, et al. Distinct patterns of brain activity in young
        carriers of the APOE-e4 allele.
        Proc Natl Acad Sci USA (PNAS), 106::7209-7214, 2009.

    [4] S.Smith, et al. Methods for network modelling from high quality rfMRI data.
        Meeting of the Organization for Human Brain Mapping. 2014

    [5] J.X. O'Reilly et al. Distinct and overlapping functional zones in the
        cerebellum defined by resting state functional connectivity.
        Cerebral Cortex, 2009.

    Note: See description for terms & conditions on data usage.

    """
    url = "http://www.nitrc.org/frs/download.php/8037/Megatrawls.tgz"
    opts = {'uncompress': True}

    error_message = "Invalid {0} input is provided: {1}, choose one of them {2}"
    # standard dataset terms
    dimensionalities = [25, 50, 100, 200, 300]
    if dimensionality not in dimensionalities:
        raise ValueError(error_message.format('dimensionality', dimensionality,
                                              dimensionalities))
    timeseries_methods = ['multiple_spatial_regression', 'eigen_regression']
    if timeseries not in timeseries_methods:
        raise ValueError(error_message.format('timeseries', timeseries,
                                              timeseries_methods))
    output_matrices_names = ['full_correlation', 'partial_correlation']
    if matrices not in output_matrices_names:
        raise ValueError(error_message.format('matrices', matrices,
                                              output_matrices_names))

    dataset_name = 'Megatrawls'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    description = _get_dataset_descr(dataset_name)

    timeseries_map = dict(multiple_spatial_regression='ts2', eigen_regression='ts3')
    matrices_map = dict(full_correlation='Znet1.txt', partial_correlation='Znet2.txt')
    filepath = [(os.path.join(
        '3T_Q1-Q6related468_MSMsulc_d%d_%s' % (dimensionality, timeseries_map[timeseries]),
        matrices_map[matrices]), url, opts)]

    # Fetch all the files
    files = _fetch_files(data_dir, filepath, resume=resume, verbose=verbose)

    # Load the files into arrays
    correlation_matrices = csv_to_array(files[0])

    return Bunch(
        dimensions=dimensionality,
        timeseries=timeseries,
        matrices=matrices,
        correlation_matrices=correlation_matrices,
        description=description)


def fetch_cobre(n_subjects=10, data_dir=None, url=None, verbose=1):
    """Fetch COBRE datasets preprocessed using NIAK 0.12.4 pipeline.

    Downloads and returns preprocessed resting state fMRI datasets and
    phenotypic information such as demographic, clinical variables,
    measure of frame displacement FD (an average FD for all the time
    frames left after censoring).

    For each subject, this function also returns .mat files which contains
    all the covariates that have been regressed out of the functional data.
    The covariates such as motion parameters, mean CSF signal, etc. It also
    contains a list of time frames that have been removed from the time series
    by censoring for high motion.

    NOTE: The number of time samples vary, as some samples have been removed
    if tagged with excessive motion. This means that data is already time
    filtered. See output variable 'description' for more details.

    .. versionadded 0.2.3

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load from maximum of 146 subjects.
        By default, 10 subjects will be loaded. If n_subjects=None,
        all subjects will be loaded.

    data_dir: str, optional
        Path to the data directory. Used to force data storage in a
        specified location. Default: None

    url: str, optional
        Override download url. Used for test only (or if you setup a
        mirror of the data). Default: None

    verbose: int, optional
       Verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the attributes are:

        - 'func': string list
            Paths to Nifti images.
        - 'mat_files': string list
            Paths to .mat files of each subject.
        - 'phenotypic': ndarray
            Contains data of clinical variables, sex, age, FD.
        - 'description': data description of the release and references.

    Notes
    -----
    More information about datasets structure, See:
    https://figshare.com/articles/COBRE_preprocessed_with_NIAK_0_12_4/1160600
    """
    if url is None:
        # Here we use the file that provides URL for all others
        url = "https://figshare.com/api/articles/1160600/15/files"

    dataset_name = 'cobre'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    fdescr = _get_dataset_descr(dataset_name)

    # First, fetch the file that references all individual URLs
    files = _fetch_files(data_dir,
                         [("files", url + "?offset=0&limit=300", {})],
                         verbose=verbose)[0]
    files = json.load(open(files, 'r'))
    # Index files by name
    files_ = {}
    for f in files:
        files_[f['name']] = f
    files = files_

    # Fetch the phenotypic file and load it
    csv_name = 'cobre_model_group.csv'
    csv_file = _fetch_files(
        data_dir, [(csv_name, files[csv_name]['downloadUrl'],
                    {'md5': files[csv_name].get('md5', None),
                     'move': csv_name})],
        verbose=verbose)[0]

    # Load file in filename to numpy arrays
    names = ['id', 'sz', 'age', 'sex', 'fd']
    csv_array = np.recfromcsv(csv_file, names=names, skip_header=True)
    # Change dtype of id and condition column
    csv_array = csv_array.astype(
        [('id', '|U17'),
         ('sz', '<i8'),
         ('age', '<f8'),
         ('sex', '<i8'),
         ('fd', '<f8')])
    csv_array['id'] = np.char.strip(csv_array['id'], '" ')

    # Check number of subjects
    max_subjects = len(csv_array)
    if n_subjects is None:
        n_subjects = max_subjects

    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects

    n_sz = np.ceil(float(n_subjects) / max_subjects * csv_array['sz'].sum())
    n_ct = np.floor(float(n_subjects) / max_subjects *
                    np.logical_not(csv_array['sz']).sum())

    # First, restrict the csv files to the adequate number of subjects
    sz_ids = csv_array[csv_array['sz'] == 1.]['id'][:n_sz]
    ct_ids = csv_array[csv_array['sz'] == 0.]['id'][:n_ct]
    ids = np.hstack([sz_ids, ct_ids])
    csv_array = csv_array[np.in1d(csv_array['id'], ids)]

    # Call fetch_files once per subject.
    func = []
    mat = []
    for i in ids:
        f = 'fmri_' + i + '_session1_run1.nii.gz'
        m = 'fmri_' + i + '_session1_run1_extra.mat'
        f, m = _fetch_files(
            data_dir,
            [(f, files[f]['downloadUrl'], {'md5': files[f].get('md5', None),
                                           'move': f}),
             (m, files[m]['downloadUrl'], {'md5': files[m].get('md5', None),
                                           'move': m})
             ],
            verbose=verbose)
        func.append(f)
        mat.append(m)

    return Bunch(func=func, mat_files=mat, phenotypic=csv_array,
                 description=fdescr)


def _build_nv_url(base_url, filts=None):
    """Build a NeuroVault URL with the given filters.

    Parameters
    ----------
    base_url: string
        NeuroVault URL (for collections, images, etc.)

    filts: object, optional
        If filts is a dict, then key-value pairs are added to the
        querystring of the URL. Otherwise, it is ignored.
    """
    if filts and isinstance(filts, dict):
        url = '?'.join(base_url,
                       '&'.join(['='.join(it) for it in filts.items()]))
    else:
        url = base_url
    return url


def _get_nv_json(url, local_file=None, overwrite=False,
                 verbose=2):
    """Download NeuroVault json metadata; load/save locally if local_file.

    Parameters
    ----------
    url: string
        URL to download the metadata from.

    local_file: string, optional
        Path to store the downloaded metadata to.

    overwrite: bool, optional
        If True, will re-download the data, even if it has been
        previously downloaded.

    verbose: int, optional
        Defines the level of verbosity of the output.
        """
    opts = dict(overwrite=overwrite)

    if not local_file:
        # Didnt' request to save locally, but _fetch_files
        #  requires it. So, dump to a temp location.
        import tempfile
        fp, filepath = tempfile.mkstemp()
        data_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        os.close(fp)  # Avoid any potential conflict
        opts['overwrite'] = True
        opts['move'] = filepath
    else:
        data_dir = os.path.dirname(local_file)
        filename = os.path.basename(local_file)
        opts['move'] = local_file  # make sure

    fil = _fetch_files(data_dir=data_dir,
                       files=[(filename, url, opts)],
                       verbose=verbose)  # necessary for proper url print
    with io.open(fil[0], 'r', encoding='utf8') as fp:
        meta = json.load(fp)

    # Cleanup
    if local_file is None:
        os.remove(os.path.join(data_dir, filename))
    return meta


def _get_nv_collections_json(url, data_dir, overwrite=False, query_server=True,
                             verbose=2):
    """Get remote list of collections (don't cache locally).

    If offline, aggregate collections metadata from directories.
    Result is unfiltered.

    Parameters
    ----------
    url: string
        URL to download the metadata from.

    data_dir: string
        Path to store the downloaded metadata to.

    overwrite: bool, optional
        If True, will re-download the data, even if it has been
        previously downloaded.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    verbose: int, optional
        Defines the level of verbosity of the output.
    """
    try:
        if query_server:  # Fall through to cached copies below, if not.
            return _get_nv_json(url, overwrite=overwrite, verbose=verbose)
    except (_urllib.error.URLError, _urllib.error.HTTPError) as ue:
        if ue.reason[0] != 8:  # connection error
            raise
        elif overwrite:  # must requery... fail.
            raise
        print("Working offline...")

    except (_http.client.BadStatusLine):
        if overwrite:
            raise
        print("Working offline...")

    # Sort collections, so same order is achieved online & offline
    collection_dirs = [os.path.basename(p)
                       for p in glob.glob(os.path.join(data_dir, '*'))
                       if os.path.isdir(p)]
    collection_dirs = sorted(collection_dirs,
                             lambda k1, k2: int(k1) - int(k2))

    coll_meta = dict(results=[], next=None)
    for cdir in collection_dirs:
        coll_meta_path = os.path.join(data_dir, cdir,
                                      'collection_metadata.json')
        if os.path.exists(coll_meta_path):
            with io.open(coll_meta_path, 'r', encoding='utf8') as fp:
                coll_meta['results'].append(json.load(fp))
    return coll_meta


def _filter_nv_results(results, filts):
    """Filter NeuroVault metadata.

    Parameters
    ----------
    results: object
        If Iterable, then filts will be applied to each
        element.

    filts: list
        List of lambda functions that will be applied to
        each value in the list of dicts. If the lambda
        function returns False, the item is discarded.
    """
    if isinstance(filts, collections.Iterable):
        for filt in filts:
            results = [r for r in results if filt(r)]
    return results


def _fetch_nv_terms(image_ids, data_dir=None, verbose=2, query_server=True,
                    url='http://neurosynth.org/api/v2/decode/'):
    """ Grab terms for each NeuroVault image, decoded with neurosynth.

    Parameters
    ----------
    image_ids: list
        List of neurovault image IDs (int).

    data_dir: string, optional (default None)
        Path of the data directory. Used to force data storage in a specified
        location. Default: None.

    verbose: int, optional (default 0)
        Defines the level of verbosity of the output.

    url: string, optional (default None)
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    Outputs
    -------

    Dict, key: term, value: score for each image ID.

    """

    # Massage inputs
    data_dir = data_dir or _get_dataset_dir('neurosynth')
    print_frequency = (200 / verbose) if verbose else np.inf

    terms = list()
    vectorizer = DictVectorizer()
    for ii, image_id in enumerate(image_ids):
        if verbose and ii % print_frequency == 0:
            max_fetch_idx = min(ii + print_frequency, len(image_ids))
            if ii == 0:
                sys.stderr.write("Fetching terms for images (of %d)" % (
                    len(image_ids)))
            sys.stderr.write(" %d-%d" % (ii + 1, max_fetch_idx))

        # Fetch the terms
        terms_url = url + '?neurovault=%d' % image_id
        output_file = 'terms-for-image-%d.json' % image_id
        file_tuple = ((output_file, terms_url, {'move': output_file}),)
        elevations = _fetch_files(data_dir, file_tuple, verbose=verbose,
                                  query_server=query_server)[0]
        if elevations is None:
            if verbose >= 3:
                print("No terms for image %s; skipping." % image_id)
            terms.append({})
            continue

        # Read and process the terms.
        try:
            with io.open(elevations, 'r', encoding='utf8') as fp:
                data = json.load(fp)['data']
        except Exception as e:
            if verbose >= 2:
                print("Exception downloading term for image %s: %s" % (
                    image_id, e))
            if os.path.exists(elevations):
                os.remove(elevations)
            terms.append({})
        else:
            data = data['values']
            terms.append(data if data is not None else {})
    if verbose:
        sys.stderr.write(" done.\n")

    # Transform and filter the terms.
    X = vectorizer.fit_transform(terms).toarray()  # noqa
    all_terms = dict([(name, X[:, idx])
                      for name, idx in vectorizer.vocabulary_.items()])
    good_terms = dict([(t, v) for t, v in all_terms.items()
                       if np.sum(v[v > 0]) > 0.])
    return good_terms


def fetch_neurovault(max_images=np.inf,
                     query_server=True,
                     fetch_terms=False,
                     exclude_unpublished=False,
                     exclude_known_bad_images=True,
                     collection_ids=(),
                     image_ids=(), image_type=None, map_types=(),
                     collection_filters=(), image_filters=(),
                     data_dir=None, url="http://neurovault.org/api",
                     resume=True, overwrite=False, verbose=2):
    """Fetch public statistical maps from NeuroVault.org.

       Image data downloaded is matched by `collection_filters` and
       `image_filters`, if specified.

       On each request, even if data are stored locally, this function
       will requery NeuroVault for the latest colllections.
       Metadata for previously queried collections and images, as well
       as downloaded images, are cached to disk.

       Currently, no check is done to see if a collection or image has
       changed. To invalidate the cache and download new data, use the
       `resume=True` flag, with appropriate filters to limit the amount
       of metadata being downloaded.

    Parameters
    ----------
    max_images: int, optional (default np.inf)
        Maximum # of images to download from the database.
        Useful for testing out filters and analyses if downloads are slow.

    query_server: bool, optional (default: True)
        if False, then only cached data is used.

    fetch_terms: bool
        Whether to fetch terms related to each image from
        NeuroSynth.org.

    exclude_unpublished: bool, optional (default: False)
        Exclude any images that belong to a collection without a DOI.

    exclude_known_bad_images: bool
        Append filters to remove known bad collections,
        image ids, and images with parameter values that
        indicate the data are not useful.

    collection_ids: list, optional (default: None)
        A list of integer IDs of collections to search for images.
        Negative IDs are *excluded*.
        If None, all collections will be searched.

    image_ids: list, optional (default: None)
        A list of integer IDs of images to download.
        Negative IDs are *excluded*.
        If None, all images will be searched.

    image_type: string, optional (default: None)
        A string of image type to include.
        These include: "statistic_map". See the NeuroVault
        website for an update-to-date list.

    map_types: string or list, optional (default: None)
        A string, or list of strings, of map types to include.
        These include: "F map", "T map", "Z map". See the NeuroVault
        website for an update-to-date list.

    collection_filters: list or dict, optional (default None)
        Filters to limit data retrieval and return via the NeuroVault API.
        If a list, each element should be a function that
            returns True if the collection metadata is a match.
            Filtering is applied after metadata is downloaded.
        If a dict, each key-value pair is an API filter.
            The key will be checked for equality to the value.
            Keys are applied before metadata download, limiting
            the number of rows returned and possibly the number
            of round trips made.
        (see http://neurovault.org/api-docs#collapseCollections for
         API keys and collection metadata keys)

    image_filters: list or dict, optional (default None)
        Filters to limit data retrieval and return via the NeuroVault API.
        If a list, each element should be a function that
            returns True if the image metadata is a match.
            Filtering is applied after metadata is downloaded.
            Matched image metadata will trigger actual image download.
        If a dict, each key-value pair is an API filter.
            The key will be checked for equality to the value.
            Keys are applied before metadata download, limiting
            the number of rows returned and possibly the number
            of round trips made.
        (see http://neurovault.org/api-docs#collapseImages for
         API keys and collection metadata keys)

    data_dir: string, optional (default None)
        Path of the data directory. Used to force data storage in a specified
        location. Default: None.

    url: string, optional (default None)
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    overwrite: bool, optional (default False)
        If True, re-download cached image metadata and data.

    resume: bool, optional (default True)
        If True, try resuming download if possible.

    verbose: int, optional (default 0)
        Defines the level of verbosity of the output.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'collections': dict of dicts (one per collection)
            Metadata about each collection (key: collection ID)
            See http://neurovault.org/api-docs#collapseCollections
            for all available fields.
        'images': list of dicts
            Metadata of image; parallel array to func_files.
            See http://neurovault.org/api-docs#collapseImages
            for available fields.
            Also includes `local_path` (path to downloaded image)
            and `collection_id` (which indexes into `collections`)
        'func_files': list of strings
            Paths to betamaps.

    References
    ----------
    [1] Gorgolewski KJ, Varoquaux G, Rivera G, Schwartz Y, Ghosh SS, Maumet C,
        Sochat VV, Nichols TE, Poldrack RA, Poline J-B, Yarkoni T and
        Margulies DS (2015) NeuroVault.org: a web-based repository for
        collecting and sharing unthresholded statistical maps of the human
        brain. Front. Neuroinform. 9:8.
        doi: 10.3389/fninf.2015.00008
    """

    # Massage parameters, convert into image filters.
    if exclude_known_bad_images:
        bad_collects = [16]
        bad_image_ids = [
            96, 97, 98,                    # The following maps are not brain maps
            338, 339,                      # And the following are crap
            335,                           # 335 is a duplicate of 336
            3360, 3362, 3364,              # These are mean images, and not Z maps
            1202, 1163, 1931, 1101, 1099]  # Ugly / obviously not Z maps
        collection_ids = chain(collection_ids, [-bid for bid in bad_collects])
        image_ids = chain(image_ids, [-bid for bid in bad_image_ids])
        image_filters = chain(image_filters,
                              [lambda im: im.get('perc_bad_voxels', 0) < 100,
                               lambda im: im.get('brain_coverage', 100) > 0])
    if exclude_unpublished:
        collection_filters = chain(collection_filters,
                                   [lambda col: col.get('DOI') is not None])
    if collection_ids:  # positive: include; negative: exclude
        collection_ids = tuple(collection_ids)  # consume more than once
        _pos_col_ids = [cid for cid in collection_ids if cid >= 0]
        _neg_col_ids = [-cid for cid in collection_ids if cid < 0]
        if _pos_col_ids:
            collection_filters = chain(collection_filters,
                                       [lambda c: c.get('id') in _pos_col_ids])
        if _neg_col_ids:
            collection_filters = chain(collection_filters,
                                       [lambda c: c.get('id') not in _neg_col_ids])
    if image_ids:  # positive: include; negative: exclude
        image_ids = tuple(image_ids)  # consume more than once
        _pos_im_ids = [iid for iid in image_ids if iid >= 0]
        _neg_im_ids = [-iid for iid in image_ids if iid < 0]
        if _pos_im_ids:
            image_filters = chain(image_filters,
                                  [lambda im: im.get('id') in _pos_im_ids])
        if _neg_im_ids:
            image_filters = chain(image_filters,
                                  [lambda im: im.get('id') not in _neg_im_ids])
    if image_type:
        image_filters = chain(image_filters,
                              [lambda im: im.get('image_type') == image_type])
    if map_types and isinstance(map_types, _basestring):
        map_types = [map_types]
    if map_types:
        image_filters = chain(image_filters,
                              [lambda im: im.get('map_type') in map_types])
    image_filters = tuple(image_filters)  # convert to tuples, we need to consume
    collection_filters = tuple(collection_filters)  # these filters many times.
    data_dir = _get_dataset_dir('neurovault', data_dir=data_dir)

    collects = dict()
    images = []
    func_files = []

    # Retrieve the relevant collects
    collections_url = _build_nv_url(base_url=url + '/collections',
                                    filts=collection_filters)
    coll_meta = dict(next=collections_url)
    while len(func_files) < max_images and coll_meta['next'] is not None:
        # GET up to 100 collections, but without caching, and filter results.
        coll_meta = _get_nv_collections_json(coll_meta['next'],
                                             data_dir=data_dir,
                                             overwrite=overwrite,
                                             query_server=query_server,
                                             verbose=verbose)
        good_coll = _filter_nv_results(results=coll_meta['results'],
                                       filts=collection_filters)

        # Retrieve image metadata
        for coll in good_coll:
            collections_dir = os.path.join(data_dir, str(coll['id']))
            base_url = url + '/collections/%d/images' % coll['id']
            images_url = _build_nv_url(base_url=base_url,
                                       filts=image_filters)

            # Save collection metadata for all matching collections,
            #   even if no images are matched / downloaded,
            coll_meta_path = os.path.join(collections_dir,
                                          'collection_metadata.json')
            if not os.path.exists(collections_dir):
                os.makedirs(collections_dir)
            with open(coll_meta_path, 'w') as fp:
                json.dump(coll, fp)

            # Return the collections metadata for all matching collections,
            #   even if no images are matched.
            collects[coll['id']] = coll

            # Search for matching images
            imgs_meta_url = images_url
            while len(func_files) < max_images and imgs_meta_url is not None:
                prefix = re.sub('[\?=]', '_', os.path.basename(imgs_meta_url))
                filename = '%s_metadata.json' % prefix
                local_path = os.path.join(collections_dir, filename)

                tmp_meta = _get_nv_json(imgs_meta_url, local_path,
                                        overwrite=overwrite, verbose=verbose)
                all_images, imgs_meta_url = tmp_meta['results'], tmp_meta['next']

                good_images = _filter_nv_results(results=all_images,
                                                 filts=image_filters)

                # Finally, we have images to download.
                # 2. Save off collection and image metadata.
                # 3. Download the image.
                for im in good_images:
                    im_url = im['file']
                    im_filename = os.path.basename(im['file'])

                    # Download file
                    try:
                        real_image_path = _fetch_files(
                            collections_dir,
                            files=[(im_filename, im_url, {})],
                            verbose=verbose, query_server=query_server)[0]
                    except Exception as e:
                        print("ERROR: failed to download image %d (col=%d): %s" % (
                            im['id'], coll['id'], e))
                        continue
                    if real_image_path is None:
                        continue

                    # Save metadata
                    im_basename = os.path.splitext(im_filename)[0]
                    im_name = im_basename + '_metadata.json'
                    im_path = os.path.join(collections_dir, im_name)
                    with open(im_path, 'w') as fp:
                        json.dump(im, fp)

                    # Add to output struct
                    im.update(dict(collection_id=coll['id'],
                                   local_path=real_image_path))  # keep copy
                    images.append(im)
                    func_files.append(im['local_path'])

                    # Stopping criterion
                    if len(func_files) >= max_images:
                        break

    # Do term fetching after everything else.
    if fetch_terms:
        terms = _fetch_nv_terms([im['id'] for im in images],
                                query_server=query_server, verbose=verbose)
    else:
        terms = dict()

    if verbose > 0:
        print('Done.')

    # Flatten the struct
    bunch = Bunch(func_files=func_files, images=images,
                  collections=collects)
    if fetch_terms:
        bunch['terms'] = terms
    return bunch
