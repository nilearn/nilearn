"""
Utilities to download NeuroImaging datasets
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import urllib
import urllib2
import tarfile
import zipfile
import sys
import shutil
import time
import hashlib

from sklearn.datasets.base import Bunch


def _format_time(t):
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % (t)


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file
    """
    f = open(path, 'rb')
    m = hashlib.md5()
    while True:
        data = f.read(8192)
        if not data:
            break
        m.update(data)
    return m.hexdigest()


def _read_md5_sum_file(path):
    """ Reads a MD5 checksum file and returns hashes as a dictionnary
    """
    f = open(path, "r")
    hashes = {}
    while True:
        line = f.readline()
        if not line:
            break
        h, name = line.rstrip().split('  ', 1)
        hashes[name] = h
    return hashes


class ResumeURLOpener(urllib.FancyURLopener):
    """Create sub-class in order to overide error 206.  This error means a
       partial file is being sent,
       which is ok in this case.  Do nothing with this error.

       Note
       ----
       This was adapted from:
       http://code.activestate.com/recipes/83208-resuming-download-of-a-file/
    """
    def http_error_206(self, url, fp, errcode, errmsg, headers, data=None):
        pass


def _chunk_report_(bytes_so_far, total_size, t0):
    """Show downloading percentage

    Parameters
    ----------
    bytes_so_far: integer
        Number of downloaded bytes

    total_size: integer, optional
        Total size of the file. None is valid

    t0: integer, optional
        The time in seconds (as returned by time.time()) at which the
        download was started.
    """
    if total_size:
        percent = float(bytes_so_far) / total_size
        percent = round(percent * 100, 2)
        dt = time.time() - t0
        # We use a max to avoid a division by zero
        remaining = (100. - percent) / max(0.01, percent) * dt
        # Trailing whitespace is too erase extra char when message length
        # varies
        sys.stderr.write(
            "Downloaded %d of %d bytes (%0.2f%%, %s remaining)  \r"
            % (bytes_so_far, total_size, percent,
               _format_time(remaining)))
    else:
        sys.stderr.write("Downloaded %d of ? bytes\r" % (bytes_so_far))


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None,
                 initial_size=0, total_size=None, verbose=0):
    """Download a file chunk by chunk and show advancement

    Parameters
    ----------
    response: urllib.addinfourl
        Response to the download request in order to get file size

    local_file: file
        Hard disk file where data should be written

    chunk_size: integer, optional
        Size of downloaded chunks. Default: 8192

    report_hook: boolean
        Whether or not to show downloading advancement. Default: None

    initial_size: int, optional
        If resuming, indicate the initial size of the file

    Returns
    -------
    data: string
        The downloaded file.

    """
    if total_size is None:
        total_size = response.info().getheader('Content-Length').strip()
    try:
        total_size = int(total_size) + initial_size
    except Exception, e:
        if verbose > 0:
            print "Warning: total size could not be determined."
            if verbose > 1:
                print "Full stack trace: %s" % e
        total_size = None
    bytes_so_far = initial_size

    t0 = time.time()
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)

        if not chunk:
            if report_hook:
                sys.stderr.write('\n')
            break

        local_file.write(chunk)
        if report_hook:
            _chunk_report_(bytes_so_far, total_size, t0)

    return


def _get_dataset_dir(dataset_name, data_dir=None):
    """ Create if necessary and returns data directory of given dataset.

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.

    data_dir: string
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data_dir: string
        Path of the given dataset directory.

    Notes
    -----
    This function retrieve the datasets directory (or data directory) using
    the following priority :
    1. the keyword argument data_dir
    2. the environment variable NISL_DATA
    3. "nisl_data" directory into the current working directory
    """
    if not data_dir:
        data_dir = os.getenv("NISL_DATA",  os.path.join(os.getcwd(),
                             'nisl_data'))
    data_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def _uncompress_file(file, delete_archive=True):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    file: string
        path of file to be uncompressed.

    delete_archive: boolean, optional
        Wheteher or not to delete archive once it is uncompressed.
        Default: True

    Notes
    -----
    This handles zip, tar, gzip and bzip files only.
    """
    print 'extracting data from %s...' % file
    data_dir = os.path.dirname(file)
    # We first try to see if it is a zip file
    try:
        ext = os.path.splitext(file)[1]
        if ext == '.zip':
            z = zipfile.Zipfile(file)
            z.extractall(data_dir)
            z.close()
        elif ext in ['.tar', '.tgz', '.gz', '.bz2']:
            tar = tarfile.open(file, "r")
            tar.extractall(path=data_dir)
            tar.close()
        else:
            raise IOError("Uncompress: unknown file extesion: %s" % ext)
        if delete_archive:
            os.remove(file)
        print '   ...done.'
    except Exception as e:
        print 'Error uncompressing file: %s' % e
        raise


def _fetch_file(url, data_dir, resume=True, overwrite=False, md5sum=None,
                verbose=0):
    """Load requested file, downloading it if needed or requested

    Parameters
    ----------
    urls: array of strings
        Contains the urls of files to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: boolean, optional
        If true, try to resume partially downloaded files

    overwrite: boolean, optional
        If true and file already exists, delete it.

    md5sum: string, optional
        MD5 sum of the file. Checked if download of the file is required

    verbose: integer, optional
        Defines the level of verbosity of the output

    Returns
    -------
    files: array of string
        Absolute paths of downloaded files on disk

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded data are
    cleaned.
    """
    # Determine data path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_name = os.path.basename(url)
    temp_file_name = file_name + ".part"
    full_name = os.path.join(data_dir, file_name)
    temp_full_name = os.path.join(data_dir, temp_file_name)
    if os.path.exists(full_name):
        if overwrite:
            os.remove(full_name)
        else:
            return full_name
    if os.path.exists(temp_full_name):
        if overwrite:
            os.remove(temp_full_name)
    t0 = time.time()
    local_file = None
    initial_size = 0
    try:
        # Download data
        print 'Downloading data from %s ...' % url
        if resume and os.path.exists(temp_full_name):
            urlOpener = ResumeURLOpener()
            # Download has been interrupted, we try to resume it.
            local_file_size = os.path.getsize(temp_full_name)
            # If the file exists, then only download the remainder
            urlOpener.addheader("Range", "bytes=%s-" % (local_file_size))
            try:
                data = urlOpener.open(url)
            except urllib2.HTTPError:
                # There is a problem that may be due to resuming. Switch back
                # to complete download method
                return _fetch_file(url, data_dir, resume=False,
                                   overwrite=False)
            local_file = open(temp_full_name, "ab")
            initial_size = local_file_size
        else:
            data = urllib2.urlopen(url)
            local_file = open(temp_full_name, "wb")
        _chunk_read_(data, local_file, report_hook=True,
                     initial_size=initial_size, verbose=verbose)
        shutil.move(temp_full_name, full_name)
        dt = time.time() - t0
        print '...done. (%i seconds, %i min)' % (dt, dt / 60)
    except urllib2.HTTPError, e:
        print 'Error while fetching file %s.' \
            ' Dataset fetching aborted.' % file_name
        if verbose > 0:
            print "HTTP Error:", e, url
        raise
    except urllib2.URLError, e:
        print 'Error while fetching file %s.' \
            ' Dataset fetching aborted.' % file_name
        if verbose > 0:
            print "URL Error:", e, url
        raise
    finally:
        if local_file is not None:
            local_file.close()
    if md5sum is not None:
        if (_md5_sum_file(full_name) != md5sum):
            raise ValueError("File %s checksum verification has failed."
                             " Dataset fetching aborted." % local_file)
    return full_name


def _fetch_dataset(dataset_name, urls, data_dir=None, uncompress=True,
                   resume=True, folder=None, md5sums=None, verbose=0):
    """Load requested dataset, downloading it if needed or requested

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    urls: array of strings
        Contains the urls of files to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    uncompress: boolean, optional
        Ask for uncompression of the dataset. The type of the archive is
        determined automatically.

    resume: boolean, optional
        If true, try resuming download if possible

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    md5sums: dictionary, optional
        Dictionary of MD5 sums of files to download

    Returns
    -------
    files: array of string
        Absolute paths of downloaded files on disk

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded data are
    cleaned.
    """
    # Determine data path
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    if not (folder is None):
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    files = []
    for url in urls:
        try:
            md5sum = None
            file_name = os.path.basename(url)
            if md5sums is not None and file_name in md5sums:
                md5sum = md5sums[file_name]
            full_name = _fetch_file(url, data_dir, resume=resume,
                                    md5sum=md5sum, verbose=verbose)
            if uncompress:
                _uncompress_file(full_name)
            files.append(full_name)
        except Exception:
            print 'An error occured, abort fetching.' \
                ' Please see the full log above.'
            shutil.rmtree(data_dir)
            raise
    return files


def _get_dataset(dataset_name, file_names, data_dir=None, folder=None):
    """Returns absolute paths of a dataset files if exist

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    file_names: array of strings
        File that compose the dataset to be retrieved on the disk.

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: array of string
        List of dataset files on disk

    Notes
    -----
    If at least one file is missing, an IOError is raised.
    """
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    if not (folder is None):
        data_dir = os.path.join(data_dir, folder)

    file_paths = []
    for file_name in file_names:
        full_name = os.path.join(data_dir, file_name)
        if not os.path.exists(full_name):
            raise IOError("No such file: '%s'" % full_name)
        file_paths.append(full_name)
    return file_paths


###############################################################################
# Dataset downloading functions

def fetch_haxby_simple(data_dir=None, url=None, resume=True, verbose=0):
    """Download and loads an example haxby dataset

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'func': string
            Path to nifti file with bold data
        'session_target': string
            Path to text file containing session and target data
        'mask': string
            Path to nifti mask file
        'session': string
            Path to text file containing labels (can be used for
            LeaveOneLabelOut cross validation for example)

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

    # definition of dataset files
    file_names = ['attributes.txt', 'bold.nii.gz', 'mask.nii.gz',
                  'attributes_literal.txt']
    file_names = [os.path.join('pymvpa-exampledata', i) for i in file_names]

    # load the dataset
    try:
        # Try to load the dataset
        files = _get_dataset("haxby2001_simple", file_names, data_dir=data_dir)

    except IOError:
        # If the dataset does not exists, we download it
        if url is None:
            url = 'http://www.pymvpa.org/files/pymvpa_exampledata.tar.bz2'
        _fetch_dataset('haxby2001_simple', [url], data_dir=data_dir,
                           resume=resume, verbose=verbose)
        files = _get_dataset("haxby2001_simple", file_names,
                             data_dir=data_dir)

    # return the data
    return Bunch(func=files[1], session_target=files[0], mask=files[2],
                 conditions_target=files[3])


def fetch_haxby(data_dir=None, n_subjects=1, url=None, resume=True, verbose=0):
    """Download and loads complete haxby dataset

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    n_subjects: integer, optional
        Number of subjects, from 1 to 5.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'anat': string list
            Paths to anatomic images
        'func': string list
            Paths to nifti file with bold data
        'session_target': string list
            Paths to text file containing session and target data
        'mask_vt': string list
            Paths to nifti ventral temporal mask file
        'mask_face': string list
            Paths to nifti ventral temporal mask file
        'mask_house': string list
            Paths to nifti ventral temporal mask file
        'mask_face_little': string list
            Paths to nifti ventral temporal mask file
        'mask_house_little': string list
            Paths to nifti ventral temporal mask file

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

    # definition of dataset files
    file_names = ['anat.nii.gz', 'bold.nii.gz', 'labels.txt',
                  'mask4_vt.nii.gz', 'mask8b_face_vt.nii.gz',
                  'mask8b_house_vt.nii.gz', 'mask8_face_vt.nii.gz',
                  'mask8_house_vt.nii.gz']

    if n_subjects > 5:
        sys.stderr.write('Warning: there is only 5 subjects')
        n_subjects = 5

    file_names = [os.path.join('subj%d' % i, name)
                  for i in range(1, n_subjects + 1)
                  for name in file_names]

    # load the dataset
    try:
        # Try to load the dataset
        files = _get_dataset("haxby2001", file_names, data_dir=data_dir)
    except IOError:
        # If the dataset does not exists, we download it
        if url is None:
            url = 'http://data.pymvpa.org/datasets/haxby2001/'
        # Get the MD5sums file
        md5sums = _fetch_file(url + 'MD5SUMS',
                              data_dir=_get_dataset_dir("haxby2001", data_dir))
        if md5sums:
            md5sums = _read_md5_sum_file(md5sums)
        urls = ["%ssubj%d-2010.01.14.tar.gz" % (url, i)
                for i in range(1, n_subjects + 1)]
        _fetch_dataset('haxby2001', urls, data_dir=data_dir,
                       resume=resume, md5sums=md5sums, verbose=verbose)
        files = _get_dataset("haxby2001", file_names, data_dir=data_dir)

    anat = []
    func = []
    session_target = []
    mask_vt = []
    mask_face = []
    mask_house = []
    mask_face_little = []
    mask_house_little = []

    for i in range(n_subjects):
        # We are considering files 8 by 8
        i *= 8
        anat.append(files[i])
        func.append(files[i + 1])
        session_target.append(files[i + 2])
        mask_vt.append(files[i + 3])
        mask_face.append(files[i + 4])
        mask_house.append(files[i + 5])
        mask_face_little.append(files[i + 6])
        mask_house_little.append(files[i + 7])

    # return the data
    return Bunch(
        anat=anat,
        func=func,
        session_target=session_target,
        mask_vt=mask_vt,
        mask_face=mask_face,
        mask_house=mask_house,
        mask_face_little=mask_face_little,
        mask_house_little=mask_house_little)


def fetch_nyu_rest(n_subjects=None, sessions=[1], data_dir=None, verbose=0):
    """Download and loads the NYU resting-state test-retest dataset

    Parameters
    ----------
    n_subjects: integer optional
        The number of subjects to load. If None is given, all the
        subjects are used.

    n_sessions: array of integers optional
        The sessions to load. Load only the first session by default.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'func': string list
            Paths to functional images
        'anat_anon': string list
            Paths to anatomic images
        'anat_skull': string
            Paths to skull-stripped images
        'session': numpy array
            List of ids corresponding to images sessions

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
    file_names = [os.path.join('anat', 'mprage_anonymized.nii.gz'),
                  os.path.join('anat', 'mprage_skullstripped.nii.gz'),
                  os.path.join('func', 'lfo.nii.gz')]

    subjects_a = ['sub05676', 'sub08224', 'sub08889', 'sub09607', 'sub14864',
                  'sub18604', 'sub22894', 'sub27641', 'sub33259', 'sub34482',
                  'sub36678', 'sub38579', 'sub39529']
    subjects_b = ['sub45463', 'sub47000', 'sub49401', 'sub52738', 'sub55441',
                  'sub58949', 'sub60624', 'sub76987', 'sub84403', 'sub86146',
                  'sub90179', 'sub94293']

    max_subjects = len(subjects_a) + len(subjects_b)
    # Check arguments
    if n_subjects is None:
        n_subjects = len(subjects_a) + len(subjects_b)
    if n_subjects > max_subjects:
        sys.stderr.write('Warning: there is only %d subjects' % max_subjects)
        n_subjects = 25

    for i in sessions:
        if not (i in [1, 2, 3]):
            raise ValueError('NYU dataset session id must be in [1, 2, 3]')

    tars = [['1071/NYU_TRT_session1a.tar.gz', '1072/NYU_TRT_session1b.tar.gz'],
            ['1073/NYU_TRT_session2a.tar.gz', '1074/NYU_TRT_session2b.tar.gz'],
            ['1075/NYU_TRT_session3a.tar.gz', '1076/NYU_TRT_session3b.tar.gz']]

    anat_anon = []
    anat_skull = []
    func = []
    session = []
    # Loading session by session
    for session_id in sessions:
        session_path = "session" + str(session_id)
        # Load subjects in two steps, as the files are splitted
        for part in range(0, n_subjects / len(subjects_a) + 1):
            if part == 0:
                subjects = subjects_a[:min(len(subjects_a), n_subjects)]
            else:  # part == 1
                subjects = subjects_b[:min(len(subjects_b),
                                      n_subjects - len(subjects_a))]
            paths = [os.path.join(session_path, os.path.join(subject, file))
                     for subject in subjects
                     for file in file_names]
            try:
                files = _get_dataset("nyu_rest", paths, data_dir=data_dir)
            except IOError:
                url = 'http://www.nitrc.org/frs/download.php/'
                url += tars[session_id - 1][part]
                # Determine files to be downloaded
                _fetch_dataset('nyu_rest', [url], data_dir=data_dir,
                               folder=session_path, verbose=verbose)
                files = _get_dataset("nyu_rest", paths, data_dir=data_dir)
            for i in range(len(subjects)):
                # We are considering files 3 by 3
                i *= 3
                anat_anon.append(files[i])
                anat_skull.append(files[i + 1])
                func.append(files[i + 2])
                session.append(session_id)

    return Bunch(anat_anon=anat_anon, anat_skull=anat_skull, func=func,
                 session=session)


def fetch_adhd(n_subjects=None, data_dir=None, url=None, resume=True,
               verbose=0):
    """Download and loads the ADHD resting-state dataset

    Parameters
    ----------
    n_subjects: integer optional
        The number of subjects to load. If None is given, all the
        40 subjects are used.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'func': string list
            Paths to functional images
        'parameters': string list
            Parameters of preprocessing steps

    References
    ----------
    :Download:
        ftp://www.nitrc.org/fcon_1000/htdocs/indi/adhd200/sites/
            ADHD200_40sub_preprocessed.tgz

    """
    file_names = ['%s_regressors.csv', '%s_rest_tshift_RPI_voreg_mni.nii.gz']

    subjects = ['0010042', '0010128', '0023008', '0027011', '0027034',
                '1019436', '1418396', '1552181', '1679142', '2497695',
                '3007585', '3205761', '3624598', '3884955', '3994098',
                '4046678', '4164316', '6115230', '8409791', '9744150',
                '0010064', '0021019', '0023012', '0027018', '0027037',
                '1206380', '1517058', '1562298', '2014113', '2950754',
                '3154996', '3520880', '3699991', '3902469', '4016887',
                '4134561', '4275075', '7774305', '8697774', '9750701']

    max_subjects = len(subjects)
    # Check arguments
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        sys.stderr.write('Warning: there is only %d subjects' % max_subjects)
        n_subjects = max_subjects

    tars = ['ADHD200_40sub_preprocessed.tgz']

    path = os.path.join('ADHD200_40sub_preprocessed', 'data')
    func = []
    confounds = []
    subjects = subjects[:n_subjects]

    paths = [os.path.join(path, os.path.join(subject, file % subject))
             for subject in subjects
             for file in file_names]
    try:
        files = _get_dataset("adhd", paths, data_dir=data_dir)
    except IOError:
        if url is None:
            url = 'ftp://www.nitrc.org/fcon_1000/htdocs/indi/adhd200/sites/'
        url += tars[0]
        _fetch_dataset('adhd', [url], data_dir=data_dir, resume=resume,
                       verbose=verbose)
        files = _get_dataset("adhd", paths, data_dir=data_dir)
    for i in range(len(subjects)):
        # We are considering files 2 by 2
        i *= 2
        func.append(files[i + 1])
        confounds.append(files[i])

    return Bunch(func=func, confounds=confounds)
