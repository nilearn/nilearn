# -*- encoding: utf-8 -*-
"""
Utilities to download NeuroImaging datasets
"""
# Author: Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import os
import os.path
import urllib
import urllib2
import tarfile
import zipfile
import sys
import shutil
import time
import hashlib

import numpy as np
from scipy import ndimage
from sklearn.datasets.base import Bunch

import nibabel


def _format_time(t):
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % (t)


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file.
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
    """ Reads a MD5 checksum file and returns hashes as a dictionary.
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
       partial file is being sent, which is fine in this case.
       Do nothing with this error.

       Note
       ----
       This was adapted from:
       http://code.activestate.com/recipes/83208-resuming-download-of-a-file/
    """
    def http_error_206(self, url, fp, errcode, errmsg, headers, data=None):
        pass


def _chunk_report_(bytes_so_far, total_size, t0):
    """Show downloading percentage.

    Parameters
    ----------
    bytes_so_far: int
        Number of downloaded bytes

    total_size: int, optional
        Total size of the file. None is valid

    t0: int, optional
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

    chunk_size: int, optional
        Size of downloaded chunks. Default: 8192

    report_hook: bool
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


def _get_dataset_dir(dataset_name, data_dir=None, folder=None,
                     create_dir=True):
    """ Create if necessary and returns data directory of given dataset.

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    create_dir: bool, optional
        If the directory does not exist, determine whether or not it is created

    Returns
    -------
    data_dir: string
        Path of the given dataset directory.

    Notes
    -----
    This function retrieve the datasets directory (or data directory) using
    the following priority :
    1. the keyword argument data_dir
    2. the environment variable NILEARN_DATA
    3. "nilearn_data" directory into the current working directory
    """
    if not data_dir:
        data_dir = os.getenv("NILEARN_DATA", os.path.join(os.getcwd(),
                             'nilearn_data'))
    data_dir = os.path.join(data_dir, dataset_name)
    if folder is not None:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def _uncompress_file(file, delete_archive=True):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    file: string
        path of file to be uncompressed.

    delete_archive: bool, optional
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
        filename, ext = os.path.splitext(file)
        processed = False
        if ext == '.zip':
            z = zipfile.ZipFile(file)
            z.extractall(data_dir)
            z.close()
            processed = True
        elif ext == '.gz':
            import gzip
            gz = gzip.open(file)
            out = open(filename, 'wb')
            shutil.copyfileobj(gz, out, 8192)
            gz.close()
            out.close()
            # If file is .tar.gz, this will be handle in the next case
            filename, ext = os.path.splitext(filename)
            processed = True
        if ext in ['.tar', '.tgz', '.bz2']:
            tar = tarfile.open(file, "r")
            tar.extractall(path=data_dir)
            tar.close()
            processed = True
        if not processed:
            raise IOError("Uncompress: unknown file extension: %s" % ext)
        if delete_archive:
            os.remove(file)
        print '   ...done.'
    except Exception as e:
        print 'Error uncompressing file: %s' % e
        raise


def _fetch_file(url, data_dir, resume=True, overwrite=False,
                md5sum=None, verbose=0):
    """Load requested file, downloading it if needed or requested.

    Parameters
    ----------
    url: string
        Contains the url of the file to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: bool, optional
        If true, try to resume partially downloaded files

    overwrite: bool, optional
        If true and file already exists, delete it.

    md5sum: string, optional
        MD5 sum of the file. Checked if download of the file is required

    verbose: int, optional
        Defines the level of verbosity of the output

    Returns
    -------
    files: string
        Absolute path of downloaded file.

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded files are
    removed.
    """
    # Determine data path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_name = os.path.basename(url)
    # Eliminate vars if needed
    file_name = file_name.split('?')[0]
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
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
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
            if not local_file.closed:
                local_file.close()
    if md5sum is not None:
        if (_md5_sum_file(full_name) != md5sum):
            raise ValueError("File %s checksum verification has failed."
                             " Dataset fetching aborted." % local_file)
    return full_name


def _fetch_dataset(dataset_name, urls, data_dir=None, uncompress=True,
                   resume=True, folder=None, md5sums=None, verbose=0):
    """Load requested dataset, downloading it if needed or requested.

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    urls: iterable of strings
        Contains the urls of files to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    uncompress: bool, optional
        Ask for uncompression of the dataset. The type of the archive is
        determined automatically.

    resume: bool, optional
        If true, try resuming download if possible

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    md5sums: dict, optional
        Dictionary of MD5 sums of files to download

    Returns
    -------
    files: list of string
        Absolute paths of downloaded files on disk

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded files are
    removed.
    """
    # Determine data path
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, folder=folder)

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
            print ('An error occured, fetching aborted.' +
                   ' Please see the full log above.')
            shutil.rmtree(data_dir)
            raise
    return files


def _get_dataset(dataset_name, file_names, data_dir=None, folder=None):
    """Returns absolute paths of a dataset files if they exist.

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    file_names: iterable of strings
        File that compose the dataset to be retrieved on the disk.

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: list of string
        List of dataset files on disk

    Notes
    -----
    If at least one file is missing, an IOError is raised.
    """
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, folder=folder)

    file_paths = []
    for file_name in file_names:
        full_name = os.path.join(data_dir, file_name)
        if not os.path.exists(full_name):
            raise IOError("No such file: '%s'" % full_name)
        file_paths.append(full_name)
    return file_paths


###############################################################################
# Dataset downloading functions

def fetch_craddock_2011_atlas(data_dir=None, url=None, resume=True, verbose=0):
    """Download and return file names for the Craddock 2011 parcellation

    The provided images are in MNI152 space.

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    url: string
        url of file to download.

    resume: bool
        whether to resumed download of a partly-downloaded file.

    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:
        scorr_mean, tcorr_mean,
        scorr_2level, tcorr_2level,
        random

    References
    ----------
    Licence: Creative Commons Attribution Non-commercial Share Alike
    http://creativecommons.org/licenses/by-nc-sa/2.5/

    Craddock, R. Cameron, G.Andrew James, Paul E. Holtzheimer, Xiaoping P. Hu,
    and Helen S. Mayberg. "A Whole Brain fMRI Atlas Generated via Spatially
    Constrained Spectral Clustering". Human Brain Mapping 33, no 8 (2012):
    1914–1928. doi:10.1002/hbm.21333.

    See http://www.nitrc.org/projects/cluster_roi/ for more information
    on this parcellation.
    """
    dataset_name = "craddock_2011"
    keys = ("scorr_mean", "tcorr_mean",
            "scorr_2level", "tcorr_2level",
            "random")
    filenames = ["scorr05_mean_all.nii.gz", "tcorr05_mean_all.nii.gz",
                 "scorr05_2level_all.nii.gz", "tcorr05_2level_all.nii.gz",
                 "random_all.nii.gz"]

    try:
        sub_files = _get_dataset(dataset_name, filenames, data_dir=data_dir)
    except IOError:
        if url is None:
            url = "ftp://www.nitrc.org/home/groups/cluster_roi/htdocs"\
                  + "/Parcellations/craddock_2011_parcellations.tar.gz"
            _fetch_dataset(dataset_name, [url], data_dir=data_dir,
                           resume=resume, verbose=verbose)
            sub_files = _get_dataset(dataset_name,
                                     filenames, data_dir=data_dir)

    params = dict(zip(keys, sub_files))
    return Bunch(**params)


def fetch_yeo_2011_atlas(data_dir=None, url=None, resume=True, verbose=0):
    """Download and return file names for the Yeo 2011 parcellation.

    The provided images are in MNI152 space.

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    url: string
        url of file to download.

    resume: bool
        whether to resumed download of a partly-downloaded file.

    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:

        - "tight_7", "liberal_7": 7-region parcellations, resp. tightly
          fitted to cortex shape, and liberally fitted.

        - "tight_17", "liberal_17": 17-region parcellations.

        - "colors_7", "colors_17": colormaps (text files) for 7- and 17-region
          parcellation respectively.

        - "anat": anatomy image.

    Notes
    -----
    For more information on this dataset's structure, see
    http://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011

    Yeo BT, Krienen FM, Sepulcre J, Sabuncu MR, Lashkari D, Hollinshead M,
    Roffman JL, Smoller JW, Zollei L., Polimeni JR, Fischl B, Liu H,
    Buckner RL. The organization of the human cerebral cortex estimated by
    intrinsic functional connectivity. J Neurophysiol 106(3):1125-65, 2011.

    Licence: unknown.
    """
    dataset_name = "yeo_2011"
    keys = ("tight_7", "liberal_7",
            "tight_17", "liberal_17",
            "colors_7", "colors_17", "anat")
    filenames = [os.path.join("Yeo_JNeurophysiol11_MNI152", f) for f in (
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_7Networks_ColorLUT.txt",
        "Yeo2011_17Networks_ColorLUT.txt",
        "FSL_MNI152_FreeSurferConformed_1mm.nii.gz")
    ]

    try:
        sub_files = _get_dataset(dataset_name, filenames, data_dir=data_dir)
    except IOError:
        if url is None:
            url = "ftp://surfer.nmr.mgh.harvard.edu/pub/data/"\
                  "Yeo_JNeurophysiol11_MNI152.zip"
            _fetch_dataset(dataset_name, [url], data_dir=data_dir,
                           resume=resume, verbose=verbose)
            sub_files = _get_dataset(dataset_name,
                                     filenames, data_dir=data_dir)

    params = dict(zip(keys, sub_files))
    return Bunch(**params)


def fetch_icbm152_2009(data_dir=None, url=None, resume=True, verbose=0):
    """Download and load the ICBM152 template (dated 2009)

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Use to forec data storage in a non-
        standard location. Default: None (meaning: default)
    url: string, optional
        Download URL of the dataset. Overwrite the default URL.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, interest keys are:
        "t1", "t2", "t2_relax", "pd": anatomical images obtained with the
        given modality (resp. T1, T2, T2 relaxometry and proton
        density weighted). Values are file paths.
        "gm", "wm", "csf": segmented images, giving resp. gray matter,
        white matter and cerebrospinal fluid. Values are file paths.
        "eye_mask", "face_mask", "mask": use these images to mask out
        parts of mri images. Values are file paths.

    References
    ----------
    VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins
    and BDCG, "Unbiased average age-appropriate atlases for pediatric studies",
    NeuroImage,Volume 54, Issue 1, January 2011

    VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins,
    "Unbiased nonlinear average age-appropriate brain templates from birth
    to adulthood", NeuroImage, Volume 47, Supplement 1, July 2009, Page S102
    Organization for Human Brain Mapping 2009 Annual Meeting.

    DL Collins, AP Zijdenbos, WFC Baaré and AC Evans,
    "ANIMAL+INSECT: Improved Cortical Structure Segmentation",
    IPMI Lecture Notes in Computer Science, 1999, Volume 1613/1999, 210–223

    Notes
    -----
    For more information about this dataset's structure:
    http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
    """

    keys = ("csf", "gm", "wm",
            "pd", "t1", "t2", "t2_relax",
            "eye_mask", "face_mask", "mask")
    filenames = [os.path.join("mni_icbm152_nlin_sym_09a", name)
                 for name in ("mni_icbm152_csf_tal_nlin_sym_09a.nii",
                              "mni_icbm152_gm_tal_nlin_sym_09a.nii",
                              "mni_icbm152_wm_tal_nlin_sym_09a.nii",

                              "mni_icbm152_pd_tal_nlin_sym_09a.nii",
                              "mni_icbm152_t1_tal_nlin_sym_09a.nii",
                              "mni_icbm152_t2_tal_nlin_sym_09a.nii",
                              "mni_icbm152_t2_relx_tal_nlin_sym_09a.nii",

                              "mni_icbm152_t1_tal_nlin_sym_09a_eye_mask.nii",
                              "mni_icbm152_t1_tal_nlin_sym_09a_face_mask.nii",
                              "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii")]

    try:
        sub_files = _get_dataset("icbm152_2009", filenames, data_dir=data_dir)
    except IOError:
        if url is None:
            url = "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/"\
                  "mni_icbm152_nlin_sym_09a_nifti.zip"
            _fetch_dataset("icbm152_2009", [url], data_dir=data_dir,
                           resume=resume, verbose=verbose)
            sub_files = _get_dataset("icbm152_2009",
                                     filenames, data_dir=data_dir)

    params = dict(zip(keys, sub_files))
    return Bunch(**params)


def fetch_haxby_simple(data_dir=None, url=None, resume=True, verbose=0):
    """Download and load an example haxby dataset

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, interest attributes are:
        'func': string.  Path to nifti file with bold data.
        'session_target': string. Path to text file containing session and
        target data.
        'mask': string. Path to nifti mask file.
        'session': string. Path to text file containing labels (can be used
        for LeaveOneLabelOut cross validation for example).

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

    n_subjects: int, optional
        Number of subjects, from 1 to 5.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        'anat': string list. Paths to anatomic images.
        'func': string list. Paths to nifti file with bold data.
        'session_target': string list. Paths to text file containing
        session and target data.
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

    More informations about its structure :
    http://dev.pymvpa.org/datadb/haxby2001.html

    See `additional information
    <http://www.sciencemag.org/content/293/5539/2425>`
    """

    # definition of dataset files
    file_names = ['anat.nii.gz', 'bold.nii.gz', 'labels.txt',
                  'mask4_vt.nii.gz', 'mask8b_face_vt.nii.gz',
                  'mask8b_house_vt.nii.gz', 'mask8_face_vt.nii.gz',
                  'mask8_house_vt.nii.gz']

    if n_subjects > 5:
        sys.stderr.write('Warning: there are only 5 subjects')
        n_subjects = 5

    # load the dataset
    anat = []
    func = []
    session_target = []
    mask_vt = []
    mask_face = []
    mask_house = []
    mask_face_little = []
    mask_house_little = []

    md5sums = None

    for i in range(1, n_subjects + 1):
        file_paths = [os.path.join('subj%d' % i, name)
                      for name in file_names]

        try:
            # Try to load the dataset
            sub_files = _get_dataset("haxby2001", file_paths,
                                     data_dir=data_dir)
        except IOError:
            # If the dataset does not exists, we download it
            if url is None:
                url = 'http://data.pymvpa.org/datasets/haxby2001/'
            # Get the MD5sums file
            if md5sums is None:
                md5sums = _fetch_file(url + 'MD5SUMS',
                                      data_dir=_get_dataset_dir("haxby2001",
                                                                data_dir))
                if md5sums:
                    md5sums = _read_md5_sum_file(md5sums)

            _fetch_dataset('haxby2001',
                           ["%ssubj%d-2010.01.14.tar.gz" % (url, i)],
                           data_dir=data_dir, resume=resume, verbose=verbose)
            sub_files = _get_dataset("haxby2001", file_paths,
                                     data_dir=data_dir)

        anat.append(sub_files[0])
        func.append(sub_files[1])
        session_target.append(sub_files[2])
        mask_vt.append(sub_files[3])
        mask_face.append(sub_files[4])
        mask_house.append(sub_files[5])
        mask_face_little.append(sub_files[6])
        mask_house_little.append(sub_files[7])

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
    """Download and load the ADHD resting-state dataset.

    Parameters
    ----------
    n_subjects: int, optional
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
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        - 'func': string list. Paths to functional images
        - 'parameters': string list. Parameters of preprocessing steps

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


def fetch_msdl_atlas(data_dir=None, url=None, resume=True, verbose=0):
    """Download and load the MSDL brain atlas.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        - 'labels': str. Path to csv file containing labels.
        - 'maps': str. path to nifti file containing regions definition.

    References
    ----------
    :Download:
        https://team.inria.fr/parietal/files/2013/05/MSDL_rois.zip

    :Paper to cite:
        `Multi-subject dictionary learning to segment an atlas of brain
        spontaneous activity <http://hal.inria.fr/inria-00588898/en>`_
        Gaël Varoquaux, Alexandre Gramfort, Fabian Pedregosa, Vincent Michel,
        Bertrand Thirion. Information Processing in Medical Imaging, 2011,
        pp. 562-573, Lecture Notes in Computer Science.

    :Other references:
        `Learning and comparing functional connectomes across subjects
        <http://hal.inria.fr/hal-00812911/en>`_.
        Gaël Varoquaux, R.C. Craddock NeuroImage, 2013.

    """
    dataset_name = "msdl_atlas"
    file_names = ['msdl_rois_labels.csv', 'msdl_rois.nii']
    tars = ['MSDL_rois.zip']
    path = "MSDL_rois"  # created by unzipping the above archive.

    paths = [os.path.join(path, fname) for fname in file_names]

    try:
        files = _get_dataset(dataset_name, paths, data_dir=data_dir)
    except IOError:
        if url is None:
            url = 'https://team.inria.fr/parietal/files/2013/05/' + tars[0]
        _fetch_dataset(dataset_name, [url], data_dir=data_dir, resume=resume,
                       verbose=verbose)
        files = _get_dataset(dataset_name, paths, data_dir=data_dir)

    return Bunch(labels=files[0], maps=files[1])


def load_harvard_oxford(atlas_name,
                        dirname="/usr/share/data/harvard-oxford-atlases/"
                        "HarvardOxford/", symmetric_split=False):
    """Load Harvard-Oxford parcellation.

    This function does not download anything, files must all be already on
    disk. They are distributed with FSL.

    Parameters
    ==========
    atlas_name: string
        Name of atlas to load. Can be:
        cort-maxprob-thr0-1mm,  cort-maxprob-thr0-2mm,
        cort-maxprob-thr25-1mm, cort-maxprob-thr25-2mm,
        cort-maxprob-thr50-1mm, cort-maxprob-thr50-2mm,
        sub-maxprob-thr0-1mm,  sub-maxprob-thr0-2mm,
        sub-maxprob-thr25-1mm, sub-maxprob-thr25-2mm,
        sub-maxprob-thr50-1mm, sub-maxprob-thr50-2mm,
        cort-prob-1mm, cort-prob-2mm,
        sub-prob-1mm, sub-prob-2mm

    dirname: string, optional
        This is the neurodebian's directory for FSL data. It may be different
        with another distribution / installation.

    symmetric_split: bool, optional
        If True, split every symmetric region in left and right parts.
        Effectively doubles the number of regions. Default: False.
        Not implemented for probabilistic atlas (*-prob-* atlases)

    Returns
    =======
    regions: nibabel.Nifti1Image
        regions definition, as a label image.
    """
    if atlas_name not in ("cort-maxprob-thr0-1mm", "cort-maxprob-thr0-2mm",
                          "cort-maxprob-thr25-1mm", "cort-maxprob-thr25-2mm",
                          "cort-maxprob-thr50-1mm", "cort-maxprob-thr50-2mm",
                          "sub-maxprob-thr0-1mm", "sub-maxprob-thr0-2mm",
                          "sub-maxprob-thr25-1mm", "sub-maxprob-thr25-2mm",
                          "sub-maxprob-thr50-1mm", "sub-maxprob-thr50-2mm",
                          "cort-prob-1mm", "cort-prob-2mm",
                          "sub-prob-1mm", "sub-prob-2mm"
                          ):
        raise ValueError("Invalid atlas name: {0}".format(atlas_name))

    filename = os.path.join(dirname, "HarvardOxford-") + atlas_name + ".nii.gz"
    regions_img = nibabel.load(filename)

    if not symmetric_split:
        return regions_img

    if atlas_name in ("cort-prob-1mm", "cort-prob-2mm",
                      "sub-prob-1mm", "sub-prob-2mm"):
        raise ValueError("Region splitting not supported for probabilistic "
                         "atlases")

    regions = regions_img.get_data()

    labels = np.unique(regions)
    slices = ndimage.find_objects(regions)
    middle_ind = (regions.shape[0] - 1) / 2
    crosses_middle = [s.start < middle_ind and s.stop > middle_ind
             for s, _, _ in slices]

    # Split every zone crossing the median plane into two parts.
    # Assumes that the background label is zero.
    half = np.zeros(regions.shape, dtype=np.bool)
    half[:middle_ind, ...] = True
    new_label = max(labels) + 1
    # Put zeros on the median plane
    regions[middle_ind, ...] = 0
    for label, crosses in zip(labels[1:], crosses_middle):
        if not crosses:
            continue
        regions[np.logical_and(regions == label, half)] = new_label
        new_label += 1

    return nibabel.Nifti1Image(regions, regions_img.get_affine())


def fetch_miyawaki2008(data_dir=None, url=None, resume=True, verbose=0):
    """Download and loads Miyawaki et al. 2008 dataset (153MB)

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'func': string list
            Paths to nifti file with bold data
        'label': string list
            Paths to text file containing session and target data
        'mask': string
            Path to nifti general mask file

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

    # Dataset files

    # Functional MRI:
    #   * 20 random scans (usually used for training)
    #   * 12 figure scans (usually used for testing)

    file_func_figure = [os.path.join('func', 'data_figure_run%02d.nii.gz' % i)
                        for i in range(1, 13)]

    file_func_random = [os.path.join('func', 'data_random_run%02d.nii.gz' % i)
                        for i in range(1, 21)]

    # Labels, 10x10 patches, stimuli shown to the subject:
    #   * 20 random labels
    #   * 12 figure labels (letters and shapes)

    file_label_figure = [os.path.join('label',
                                      'data_figure_run%02d_label.csv' % i)
                         for i in range(1, 13)]

    file_label_random = [os.path.join('label',
                                      'data_random_run%02d_label.csv' % i)
                         for i in range(1, 21)]

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

    file_mask = [os.path.join('mask', m) for m in file_mask]

    file_names = file_func_figure + file_func_random + \
                 file_label_figure + file_label_random + \
                 file_mask

    # Load the dataset
    files = []
    try:
        # Try to load the dataset
        files = _get_dataset("miyawaki2008", file_names, data_dir=data_dir)
    except IOError:
        # If the dataset does not exists, we download it
        url = 'https://www.nitrc.org/frs/download.php' \
              '/5899/miyawaki2008.tgz?i_agree=1&download_now=1'
        _fetch_dataset('miyawaki2008', [url], data_dir=data_dir,
                           resume=resume, verbose=verbose)
        files = _get_dataset("miyawaki2008", file_names, data_dir=data_dir)

    # Return the data
    return Bunch(
        func=files[:32],
        label=files[32:64],
        mask=files[64],
        mask_roi=files[65:])
