# *- encoding: utf-8 -*-
"""
Utilities to download NeuroImaging datasets
"""
# Author: Alexandre Abraham, Philippe Gervais
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
import fnmatch
import warnings
import cPickle as pickle

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


def _uncompress_file(file_, delete_archive=True):
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
    print 'extracting data from %s...' % file_
    data_dir = os.path.dirname(file_)
    # We first try to see if it is a zip file
    try:
        filename, ext = os.path.splitext(file_)
        processed = False
        if ext == '.zip':
            z = zipfile.ZipFile(file_)
            z.extractall(path=data_dir)
            z.close()
            processed = True
        elif ext == '.gz':
            import gzip
            gz = gzip.open(file_)
            out = open(filename, 'wb')
            shutil.copyfileobj(gz, out, 8192)
            gz.close()
            out.close()
            # If file is .tar.gz, this will be handle in the next case
            if delete_archive:
                os.remove(file_)
            file_ = filename
            filename, ext = os.path.splitext(file_)
            processed = True
        if ext in ['.tar', '.tgz', '.bz2']:
            tar = tarfile.open(file_, "r")
            tar.extractall(path=data_dir)
            tar.close()
            processed = True
        if not processed:
            raise IOError("Uncompress: unknown file extension: %s" % ext)
        if delete_archive:
            os.remove(file_)
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
            url_opener = ResumeURLOpener()
            # Download has been interrupted, we try to resume it.
            local_file_size = os.path.getsize(temp_full_name)
            # If the file exists, then only download the remainder
            url_opener.addheader("Range", "bytes=%s-" % (local_file_size))
            try:
                data = url_opener.open(url)
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


def movetree(src, dst):
    """Move an entire tree to another directory. Any existing file is
    overwritten"""
    names = os.listdir(src)

    # Create destination dir if it does not exist
    if not os.path.exists(dst):
        os.makedirs(dst)
    errors = []

    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.isdir(srcname) and os.path.isdir(dstname):
                movetree(srcname, dstname)
                os.rmdir(srcname)
            else:
                shutil.move(srcname, dstname)
        except (IOError, os.error) as why:
            errors.append((srcname, dstname, str(why)))
        # catch the Error from the recursive movetree so that we can
        # continue with other files
        except Exception as err:
            errors.extend(err.args[0])
    if errors:
        raise Exception(errors)


def _fetch_files(dataset_name, files, data_dir=None, resume=True, folder=None,
                 verbose=0):
    """Load requested dataset, downloading it if needed or requested.

    If needed, _fetch_files download data in a sandbox and check that all files
    are present before moving it to their final destination. This avoids
    corruption of an existing dataset.

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    files: list of (string, string, dict)
        List of files and their corresponding url. The dictionary contains
        options regarding the files. Options supported are 'uncompress' to
        indicates that the file is an archive, 'md5sum' to check the md5 sum of
        the file and 'move' if renaming the file or moving it to a subfolder is
        needed.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: bool, optional
        If true, try resuming download if possible

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    Returns
    -------
    files: list of string
        Absolute paths of downloaded files on disk
    """
    # There are two working directories here:
    # - data_dir is the destination directory of the dataset
    # - temp_dir is a temporary directory dedicated to this fetching call. All
    #   files that must be downloaded will be in this directory. If a corrupted
    #   file is found, or a file is missing, this working directory will be
    #   deleted.
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, folder=folder)
    files_pickle = pickle.dumps(files)
    files_md5 = hashlib.md5(files_pickle).hexdigest()
    temp_dir = os.path.join(data_dir, files_md5)

    # Abortion flag, in case of error
    abort = False

    files_ = []
    for file_, url, opts in files:
        # 3 possibilities:
        # - the file exists in data_dir, nothing to do.
        # - the file does not exists: we download it in temp_dir
        # - the file exists in temp_dir: this can happen if an archive has been
        #   downloaded. There is nothing to do

        # Target file in the data_dir
        target_file = os.path.join(data_dir, file_)
        # Target file in temp dir
        temp_target_file = os.path.join(temp_dir, file_)
        if (not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)):
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            md5sum = opts.get('md5sum', None)
            dl_file = _fetch_file(url, temp_dir, resume=resume,
                                  verbose=verbose, md5sum=md5sum)
            if 'move' in opts:
                # XXX: here, move is supposed to be a dir, it can be a name
                move = os.path.join(temp_dir, opts['move'])
                move_dir = os.path.dirname(os.path.join(temp_dir, move))
                if not os.path.exists(move_dir):
                    os.makedirs(move_dir)
                shutil.move(os.path.join(temp_dir, dl_file), move)
                dl_file = os.path.join(temp_dir, opts['move'])
            if 'uncompress' in opts:
                try:
                    _uncompress_file(dl_file)
                except:
                    abort = True
        if (not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)):
            warnings.warn('An error occured while fetching %s' % file_)
            abort = True
        if abort:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise IOError('Fetching aborted. See error above')
        files_.append(target_file)
    # If needed, move files from temps directory to final directory.
    if os.path.exists(temp_dir):
        #XXX We could only moved the files requested
        #XXX Movetree can go wrong
        movetree(temp_dir, data_dir)
        shutil.rmtree(temp_dir)
    return files_


def _tree(path, pattern=None, dictionary=False):
    """ Return a directory tree under the form of a dictionaries and list

    Parameters:
    -----------
    path: string
        Path browsed

    pattern: string, optional
        Pattern used to filter files (see fnmatch)

    dictionary: boolean, optional
        If True, the function will return a dict instead of a list
    """
    files = []
    dirs = [] if not dictionary else {}
    for file_ in os.listdir(path):
        file_path = os.path.join(path, file_)
        if os.path.isdir(file_path):
            if not dictionary:
                dirs.append((file_, _tree(file_path, pattern)))
            else:
                dirs[file_] = _tree(file_path, pattern)
        else:
            if pattern is None or fnmatch.fnmatch(file_, pattern):
                files.append(file_path)
    files = sorted(files)
    if not dictionary:
        return sorted(dirs) + files
    if len(dirs) == 0:
        return files
    if len(files) > 0:
        dirs['.'] = files
    return dirs


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

    url = "ftp://www.nitrc.org/home/groups/cluster_roi/htdocs" \
          "/Parcellations/craddock_2011_parcellations.tar.gz"
    opts = {'uncompress': True}

    dataset_name = "craddock_2011"
    keys = ("scorr_mean", "tcorr_mean",
            "scorr_2level", "tcorr_2level",
            "random")
    filenames = [
            ("scorr05_mean_all.nii.gz", url, opts),
            ("tcorr05_mean_all.nii.gz", url, opts),
            ("scorr05_2level_all.nii.gz", url, opts),
            ("tcorr05_2level_all.nii.gz", url, opts),
            ("random_all.nii.gz", url, opts)
    ]

    sub_files = _fetch_files(dataset_name, filenames, data_dir=data_dir,
                             resume=resume)

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
    url = "ftp://surfer.nmr.mgh.harvard.edu/" \
          "pub/data/Yeo_JNeurophysiol11_MNI152.zip"
    opts = {'uncompress': True}

    dataset_name = "yeo_2011"
    keys = ("tight_7", "liberal_7",
            "tight_17", "liberal_17",
            "colors_7", "colors_17", "anat")
    filenames = [(os.path.join("Yeo_JNeurophysiol11_MNI152", f), url, opts)
        for f in (
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_7Networks_ColorLUT.txt",
        "Yeo2011_17Networks_ColorLUT.txt",
        "FSL_MNI152_FreeSurferConformed_1mm.nii.gz")
    ]

    sub_files = _fetch_files(dataset_name, filenames, data_dir=data_dir,
                             resume=resume)

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

    url = "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/" \
          "mni_icbm152_nlin_sym_09a_nifti.zip"
    opts = {'uncompress': True}

    keys = ("csf", "gm", "wm",
            "pd", "t1", "t2", "t2_relax",
            "eye_mask", "face_mask", "mask")
    filenames = [(os.path.join("mni_icbm152_nlin_sym_09a", name), url, opts)
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

    sub_files = _fetch_files('icbm152_2009', filenames, data_dir=data_dir,
                             resume=resume)

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

    files = _fetch_files('haxby2001_simple', files, data_dir=data_dir,
                         resume=resume)

    # return the data
    return Bunch(func=files[1], session_target=files[0], mask=files[2],
                 conditions_target=files[3])


def fetch_haxby(data_dir=None, n_subjects=1, fetch_stimuli=False,
                url=None, resume=True, verbose=0):
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

    if n_subjects > 6:
        warnings.warn('Warning: there are only 6 subjects')
        n_subjects = 6

    # Dataset files
    url = 'http://data.pymvpa.org/datasets/haxby2001/'
    md5sums = _fetch_files("haxby2001", [('MD5SUMS', url + 'MD5SUMS', {})],
                           data_dir=data_dir)[0]
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

    files = _fetch_files('haxby2001', files, data_dir=data_dir,
                         resume=resume)

    if n_subjects == 6:
        files.append(None)  # None value because subject 6 has no anat

    kwargs = {}
    if fetch_stimuli:
        readme = _fetch_files('haxby2001',
                [(os.path.join('stimuli', 'README'),
                  url + 'stimuli-2010.01.14.tar.gz', {'uncompress': True})],
                data_dir=data_dir, resume=resume)[0]
        kwargs['stimuli'] = _tree(os.path.dirname(readme), pattern='*.jpg',
                                  dictionary=True)

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
            **kwargs)


def fetch_nyu_rest(n_subjects=None, sessions=[1], data_dir=None, resume=True,
                   verbose=0):
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
        n_subjects = 25

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

    anat_anon = _fetch_files('nyu_rest', anat_anon, resume=resume,
                             data_dir=data_dir)
    anat_skull = _fetch_files('nyu_rest', anat_skull, resume=resume,
                              data_dir=data_dir)
    func = _fetch_files('nyu_rest', func, resume=resume,
                        data_dir=data_dir)

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
        ftp://www.nitrc.org/fcon_1000/htdocs/indi/adhd200/sites/ADHD200_40sub_preprocessed.tgz

    """

    if url is None:
        url = 'http://connectir.projects.nitrc.org'
    f1 = url + '/adhd40_p1.tar.gz'
    f2 = url + '/adhd40_p2.tar.gz'
    f3 = url + '/adhd40_p3.tar.gz'
    f4 = url + '/adhd40_p4.tar.gz'
    f1_opts = {'uncompress': True}
    f2_opts = {'uncompress': True}
    f3_opts = {'uncompress': True}
    f4_opts = {'uncompress': True}

    fname = '%s_rest_tshift_RPI_voreg_mni.nii.gz'
    rname = '%s_regressors.csv'

    # Subjects ID per file
    sub1 = ['3902469', '7774305', '3699991']
    sub2 = ['2014113', '4275075', '1019436', '3154996', '3884955', '0027034',
            '4134561', '0027018', '6115230', '0027037', '8409791', '0027011']
    sub3 = ['3007585', '8697774', '9750701', '0010064', '0021019', '0010042',
            '0010128', '2497695', '4164316', '1552181', '4046678', '0023012']
    sub4 = ['1679142', '1206380', '0023008', '4016887', '1418396', '2950754',
            '3994098', '3520880', '1517058', '9744150', '1562298', '3205761',
            '3624598']
    subs = sub1 + sub2 + sub3 + sub4

    subjects_funcs = \
        [(os.path.join('data', i, fname % i), f1, f1_opts) for i in sub1] + \
        [(os.path.join('data', i, fname % i), f2, f2_opts) for i in sub2] + \
        [(os.path.join('data', i, fname % i), f3, f3_opts) for i in sub3] + \
        [(os.path.join('data', i, fname % i), f4, f4_opts) for i in sub4]

    subjects_confounds = \
        [(os.path.join('data', i, rname % i), f1, f1_opts) for i in sub1] + \
        [(os.path.join('data', i, rname % i), f2, f2_opts) for i in sub2] + \
        [(os.path.join('data', i, rname % i), f3, f3_opts) for i in sub3] + \
        [(os.path.join('data', i, rname % i), f4, f4_opts) for i in sub4]

    phenotypic = [('ADHD200_40subs_motion_parameters_and_phenotypics.csv', f1,
        f1_opts)]

    max_subjects = len(subjects_funcs)
    # Check arguments
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects

    subs = subs[:n_subjects]
    subjects_funcs = subjects_funcs[:n_subjects]
    subjects_confounds = subjects_confounds[:n_subjects]

    subjects_funcs = _fetch_files('adhd', subjects_funcs,
            data_dir=data_dir, resume=resume)
    subjects_confounds = _fetch_files('adhd', subjects_confounds,
            data_dir=data_dir, resume=resume)
    phenotypic = _fetch_files('adhd', phenotypic,
            data_dir=data_dir, resume=resume)[0]

    # Load phenotypic data
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',',
                               dtype=None)
    # Keep phenotypic information for selected subjects
    isubs = np.asarray(subs, dtype=int)
    phenotypic = phenotypic[[np.where(phenotypic['Subject'] == i)[0][0]
                             for i in isubs]]

    return Bunch(func=subjects_funcs, confounds=subjects_confounds,
                 phenotypic=phenotypic)


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
    url = 'https://team.inria.fr/parietal/files/2013/05/MSDL_rois.zip'
    opts = {'uncompress': True}

    dataset_name = "msdl_atlas"
    files = [(os.path.join('MSDL_rois', 'msdl_rois_labels.csv'), url, opts),
             (os.path.join('MSDL_rois', 'msdl_rois.nii'), url, opts)]

    files = _fetch_files(dataset_name, files, data_dir=data_dir,
                         resume=resume)

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

    # Load atlas name
    if atlas_name[0] == 'c':
        name_map = os.path.join(dirname, '..', 'HarvardOxford-Cortical.xml')
    else:
        name_map = os.path.join(dirname, '..', 'HarvardOxford-SubCortical.xml')
    names = {}
    from lxml import etree
    names[0] = 'Background'
    for label in etree.parse(name_map).findall('.//label'):
        names[int(label.get('index')) + 1] = label.text
    names = np.asarray(names.values())

    if not symmetric_split:
        return regions_img, names

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

    # Duplicate labels for right and left
    new_names = [names[0]]
    for n in names[1:]:
        new_names.append(n + ', right part')
    for n in names[1:]:
        new_names.append(n + ', left part')

    return nibabel.Nifti1Image(regions, regions_img.get_affine()), new_names


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

    url = 'https://www.nitrc.org/frs/download.php' \
          '/5899/miyawaki2008.tgz?i_agree=1&download_now=1'
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

    files = _fetch_files('miyawaki2008', file_names, resume=resume,
                         data_dir=data_dir)

    # Return the data
    return Bunch(
        func=files[:32],
        label=files[32:64],
        mask=files[64],
        mask_roi=files[65:])


def fetch_localizer_contrasts(contrasts, n_subjects=None, get_tmaps=False,
                              get_masks=False, get_anats=False,
                              data_dir=None, url=None, resume=True, verbose=0):
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
        Allowed values are:
        {"computation audio", ..., "sentences video"}
        or equivalently:
        {"auditory calculation", ..., "visual sentences vs checkerboard"}.

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

    verbose: int
        Verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'cmaps': string list
            Paths to nifti contrast maps
        'tmaps' string list (if 'get_tmaps' set to True)
            Paths to nifti t maps
        'masks': string list
            Paths to nifti files corresponding to the subjects individual masks
        'anats': string
            Path to nifti files corresponding to the subjects structural images

    References
    ----------
    [1] Pinel, Philippe, et al.
        "Fast reproducible identification and large-scale databasing of
         individual functional cognitive networks."
        BMC neuroscience 8.1 (2007): 91.

    Caveats
    -------
    When n_subjects < 94 is specified, all subjects are actually downloaded.
    However, the caching system only checks for the presence of the n_subjects
    first subjects on disk.

    """
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
        # Instructions
        "auditory instructions": "auditory processing",
        "visual instructions": "visual processing",
        "visual vs auditory instructions":
            "visual processing vs auditory processing",
        "auditory vs visual instructions":
            "auditory processing vs visual processing",
        "visual instructions vs checkerboard":
            "visual processing vs checkerboard",
        "instructions vs button press": "cognitive processing vs motor",
        # Sentences
        "sentences": "auditory&visual sentences",
        "sentence listening": "auditory sentences",
        "sentence reading": "visual sentences",
        "sentence reading vs checkerboard": "visual sentences vs checkerboard",
        # Calculation
        "calculation (auditory cue)": "auditory calculation",
        "calculation (visual cue)": "visual calculation",
        "calculation": "auditory&visual calculation",
        "calculation (auditory cue) vs sentences listening":
            "auditory calculation vs auditory sentences",
        "calculation (visual cue) vs sentence reading":
            "visual calculation vs sentences",
        "calculation vs sentences": "auditory&visual calculation vs sentences",
        # Button press
        "left button press (auditory)": "left auditory click",
        "left button press (visual cue)": "left visual click",
        "left button press": "left auditory&visual click",
        "left vs right button press": "left auditory & visual click vs "
           + "right auditory&visual click",
        "right button press (auditory cue)": "right auditory click",
        "right button press (visual cue)": "right visual click",
        "right button press": "right auditory&visual click",
        "right vs left button press": "right auditory & visual click "
           + "vs left auditory&visual click",
        "button press (auditory cue) vs sentences listening":
            "auditory click vs auditory sentences",
        "button press (visual cue) vs sentences reading":
            "visual click vs visual sentences",
        "button press vs instructions":
            "auditory&visual motor vs cognitive processing"}
    allowed_contrasts = contrast_name_wrapper.values()
    # convert contrast names
    contrasts_wrapped = []
    for contrast in contrasts:
        if contrast in allowed_contrasts:
            contrasts_wrapped.append(contrast)
        elif contrast in contrast_name_wrapper:
            contrasts_wrapped.append(contrast_name_wrapper[contrast])
        else:
            raise ValueError("Contrast \'%s\' is not available" % contrast)

    # It is better to perform several small requests than a big one because:
    # - Brainomics server has no cache (can lead to timeout while the archive
    #   is generated on the remote server)
    # - Local (cached) version of the files can be checked for each contrast
    opts = {'uncompress': True}
    data_types = ["c map"]
    if get_tmaps:
        data_types.append(["t map"])
    rql_types = str.join(", ", ["\"" + x + "\"" for x in data_types])
    urls = ["http://brainomics.cea.fr/localizer/brainomics_data.zip?rql="
            + urllib.quote("Any X,XT,XL,XI,XF,XD WHERE X is Scan, X type XT, "
                           "X label XL, X identifier XI, "
                           "X format XF, X description XD, "
                           "X type IN(%s), X label \"%s\"" % (rql_types, c),
                           safe=',()')
            + "&vid=data-zip"
            for c in contrasts_wrapped]
    filenames = []
    for s in np.arange(1, n_subjects + 1):
        for data_type in data_types:
            for contrast_id, contrast in enumerate(contrasts_wrapped):
                file_path = os.path.join(
                    "brainomics_data", "S%02d" % s,
                    str.replace(str.join('_', [data_type, c]), ' ', '_')
                    + ".nii.gz")
                file_tarball_url = urls[contrast_id]
                filenames.append((file_path, file_tarball_url, opts))
    # Fetch masks if asked by user
    if get_masks:
        urls.append("http://brainomics.cea.fr/localizer/brainomics_data.zip"
                    "?rql="
                    + urllib.quote("Any X,XT,XL,XI,XF,XD WHERE X is Scan, "
                                   "X type XT, X label XL, X identifier XI, "
                                   "X format XF, X description XD, "
                                   "X type IN(\"boolean mask\"), "
                                   "X label \"mask\"", safe=',()')
                    + "&vid=data-zip")
        for s in np.arange(1, n_subjects + 1):  # 94 subjects available
            file_path = os.path.join(
                "brainomics_data", "S%02d" % s, "boolean_mask.nii.gz")
            file_tarball_url = urls[-1]
            filenames.append((file_path, file_tarball_url, opts))
    # Fetch anats if asked by user
    if get_anats:
        urls.append("http://brainomics.cea.fr/localizer/brainomics_data.zip"
                    "?rql="
                    + urllib.quote("Any X,XT,XL,XI,XF,XD WHERE X is Scan, "
                                   "X type XT, X label XL, X identifier XI, "
                                   "X format XF, X description XD, "
                                   "X type IN(\"normalized T1\"), "
                                   "X label \"anatomy\"", safe=',()')
                    + "&vid=data-zip")
        for s in np.arange(1, n_subjects + 1):
            file_path = os.path.join(
                "brainomics_data", "S%02d" % s,
                "normalized_T1_anat_defaced.nii.gz")
            file_tarball_url = urls[-1]
            filenames.append((file_path, file_tarball_url, opts))

    # Actual data fetching
    files = _fetch_files('brainomics_localizer', filenames, data_dir=data_dir)
    anats = None
    masks = None
    tmaps = None
    if get_anats:
        anats = files[-n_subjects:]
        files = files[:-n_subjects]
    if get_masks:
        masks = files[-n_subjects:]
        files = files[:-n_subjects]
    if get_tmaps:
        tmaps = files(slice(1, len(files) + 1, 2))
        files = files(slice(0, len(files), 2))
    return Bunch(cmaps=files, tmaps=tmaps, mask=masks, anats=anats)


def fetch_localizer_one_structural_image(data_dir=None):
    """Download and load one single structural image from the localizer.

    Useful for plots and examples, where the structural image can serve as
    a background image.
    Subject S53 is used, but this is an arbitrary choice.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'anat': string
            Path to nifti file corresponding to structural image

    """
    opts = {'uncompress': True}
    url = ("http://brainomics.cea.fr/localizer/scan/1064245"
           "/brainomics_data.zip?rql="
           + urllib.quote("Any X,AA,AB,AC,AD,AE,AF WHERE X is Scan, "
                          "X filepath AA, X type AB, X label AC, "
                          "X format AD, X description AE, "
                          "X modification_date AF, X eid 1064245", safe=',()')
           + "&vid=data-zip")
    filename = [(os.path.join("brainomics_data", "S53",
                              "normalized_T1_anat_defaced.nii.gz"),
                 url,
                 opts)]
    # Actual data fetching
    anat = _fetch_files('brainomics_localizer', filename, data_dir=data_dir)
    return Bunch(anat=anat)


def fetch_localizer_computation_vs_sentences_task(n_subjects=None,
                                                  data_dir=None):
    """Fetch computation-sentences task contrast maps from the localizer.

    This function is only a caller for the fetch_localizer_contrasts in order
    to simplify examples reading and understanding.

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load. If None is given,
        all 94 subjects are used.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'cmaps': string list
            Paths to nifti contrast maps

    """
    res = fetch_localizer_contrasts(["computation-sentences"],
                                    n_subjects=n_subjects,
                                    get_tmaps=False, get_masks=False,
                                    get_anats=False, data_dir=data_dir,
                                    url=None, resume=True, verbose=0)
    anat = fetch_localizer_one_structural_image(data_dir=data_dir)
    res.update(anat)
    return res
