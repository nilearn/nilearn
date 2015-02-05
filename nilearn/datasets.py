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
import cStringIO as StringIO
import re
import collections

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


def _chunk_report_(bytes_so_far, total_size, initial_size, t0):
    """Show downloading percentage.

    Parameters
    ----------
    bytes_so_far: int
        Number of downloaded bytes

    total_size: int
        Total size of the file (may be 0/None, depending on download method).

    t0: int
        The time in seconds (as returned by time.time()) at which the
        download was resumed / started.

    initial_size: int
        If resuming, indicate the initial size of the file.
        If not resuming, set to zero.
    """

    if not total_size:
        sys.stderr.write("Downloaded %d of ? bytes\r" % (bytes_so_far))

    else:
        # Estimate remaining download time
        total_percent = float(bytes_so_far) / total_size

        current_download_size = bytes_so_far - initial_size
        bytes_remaining = total_size - bytes_so_far
        dt = time.time() - t0
        download_rate = current_download_size / float(dt)
        # Minimum rate of 0.01 bytes/s, to avoid dividing by zero.
        time_remaining = bytes_remaining / max(0.01, download_rate)

        # Trailing whitespace is to erase extra char when message length
        # varies
        sys.stderr.write(
            "Downloaded %d of %d bytes (%0.2f%%, %s remaining)  \r"
            % (bytes_so_far, total_size, total_percent * 100,
               _format_time(time_remaining)))


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None,
                 initial_size=0, total_size=None, verbose=1):
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

    total_size: int, optional
        Expected final size of download (None means it is unknown).

    verbose: int, optional
        verbosity level (0 means no message).

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
        if verbose > 1:
            print "Warning: total size could not be determined."
            if verbose > 2:
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
            _chunk_report_(bytes_so_far, total_size, initial_size, t0)

    return


def _get_dataset_dir(dataset_name, data_dir=None, lookup_defaults=False,
                     verbose=1):
    """ Create if necessary and returns data directory of given dataset.

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    lookup_defaults: boolean, optional
        If data_dir is not None, indicates if defaults directory should still
        be searched.

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data_dir: string
        Path of the given dataset directory.

    Notes
    -----
    This function retrieves the datasets directory (or data directory) using
    the following priority :
    1. the keyword argument data_dir
    2. the global environment variable NILEARN_SHARED_DATA
    3. the user environment variable NILEARN_DATA
    4. nilearn_data in the user home folder
    """
    # We build an array of successive paths by priority
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths = data_dir.split(':')
    if lookup_defaults or data_dir is None:
        global_data = os.getenv('NILEARN_SHARED_DATA')
        if global_data is not None:
            paths.extend(global_data.split(':'))

        local_data = os.getenv('NILEARN_DATA')
        if local_data is not None:
            paths.extend(local_data.split(':'))

        paths.append(os.path.expanduser('~/nilearn_data'))

    if verbose > 2:
        print 'Dataset search paths:', paths

    # Check if the dataset exists somewhere
    for path in paths:
        path = os.path.join(path, dataset_name)
        if os.path.exists(path) and os.path.isdir(path):
            if verbose > 1:
                print 'Dataset found in', path
            return path

    # If not, create a folder in the first writeable directory
    errors = []
    for path in paths:
        path = os.path.join(path, dataset_name)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                if verbose > 0:
                    print 'Dataset created in', path
                return path
            except Exception as exc:
                short_error_message = getattr(exc, 'strerror', str(exc))
                errors.append('\n -{0} ({1})'.format(
                    path, short_error_message))

    raise OSError('Nilearn tried to store the dataset in the following '
            'directories, but:' + ''.join(errors))


def _uncompress_file(file_, delete_archive=True, verbose=1):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    file: string
        path of file to be uncompressed.

    delete_archive: bool, optional
        Wheteher or not to delete archive once it is uncompressed.
        Default: True

    verbose: int, optional
        verbosity level (0 means no message).

    Notes
    -----
    This handles zip, tar, gzip and bzip files only.
    """
    if verbose > 0:
        print 'extracting data from %s...' % file_
    data_dir = os.path.dirname(file_)
    # We first try to see if it is a zip file
    try:
        filename, ext = os.path.splitext(file_)
        fd = open(file_, "rb")
        header = fd.read(4)
        fd.close()
        processed = False
        if zipfile.is_zipfile(file_):
            z = zipfile.ZipFile(file_)
            z.extractall(data_dir)
            z.close()
            processed = True
        elif ext == '.gz' or header.startswith('\x1f\x8b'):
            import gzip
            gz = gzip.open(file_)
            if ext == '.tgz':
                filename = filename + '.tar'
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
        if tarfile.is_tarfile(file_):
            tar = tarfile.open(file_, "r")
            tar.extractall(path=data_dir)
            tar.close()
            processed = True
        if not processed:
            raise IOError(
                    "[Uncompress] unknown archive file format: %s" % file_)
        if delete_archive:
            os.remove(file_)
        if verbose > 0:
            print '   ...done.'
    except Exception as e:
        if verbose > 0:
            print 'Error uncompressing file: %s' % e
        raise


def _filter_column(array, col, criteria):
    """ Return index array matching criteria

    Parameters
    ----------

    array: numpy array with columns
        Array in which data will be filtered

    col: string
        Name of the column

    criteria: integer (or float), pair of integers, string or list of these
        if integer, select elements in column matching integer
        if a tuple, select elements between the limits given by the tuple
        if a string, select elements that match the string
    """
    # Raise an error if the column does not exist. This is the only way to
    # test it across all possible types (pandas, recarray...)
    try:
        array[col]
    except:
        raise KeyError('Filtering criterion %s does not exist' % col)

    if not isinstance(criteria, basestring) and \
            not isinstance(criteria, tuple) and \
            isinstance(criteria, collections.Iterable):
        filter = np.zeros(array.shape[0], dtype=np.bool)
        for criterion in criteria:
            filter = np.logical_or(filter,
                        _filter_column(array, col, criterion))
        return filter

    if isinstance(criteria, tuple):
        if len(criteria) != 2:
            raise ValueError("An interval must have 2 values")
        if criteria[0] is None:
            return array[col] <= criteria[1]
        if criteria[1] is None:
            return array[col] >= criteria[0]
        filter = array[col] <= criteria[1]
        return np.logical_and(filter, array[col] >= criteria[0])

    return array[col] == criteria


def _filter_columns(array, filters, combination='and'):
    """ Return indices of recarray entries that match criteria.

    Parameters
    ----------

    array: numpy array with columns
        Array in which data will be filtered

    filters: list of criteria
        See _filter_column

    combination: string, optional
        String describing the combination operator. Possible values are "and"
        and "or".
    """
    if combination == 'and':
        fcomb = np.logical_and
        mask = np.ones(array.shape[0], dtype=np.bool)
    elif combination == 'or':
        fcomb = np.logical_or
        mask = np.zeros(array.shape[0], dtype=np.bool)
    else:
        raise ValueError('Combination mode not known: %s' % combination)

    for column in filters:
        mask = fcomb(mask, _filter_column(array, column, filters[column]))
    return mask


def _fetch_file(url, data_dir, resume=True, overwrite=False,
                md5sum=None, verbose=1):
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
        verbosity level (0 means no message).

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

    # Determine filename using URL
    parse = urllib2.urlparse.urlparse(url)
    file_name = os.path.basename(parse.path)

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
        if verbose > 0:
            displayed_url = urllib.splitquery(url)[0] if verbose == 1 else url
            print 'Dowloading data from %s ...' % displayed_url
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
                                   overwrite=False, verbose=verbose)
            local_file = open(temp_full_name, "ab")
            initial_size = local_file_size
        else:
            data = urllib2.urlopen(url)
            local_file = open(temp_full_name, "wb")
        _chunk_read_(data, local_file, report_hook=(verbose > 0),
                     initial_size=initial_size, verbose=verbose)
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        shutil.move(temp_full_name, full_name)
        dt = time.time() - t0
        if verbose > 0:
            print '...done. (%i seconds, %i min)' % (dt, dt / 60)
    except urllib2.HTTPError, e:
        if verbose > 0:
            print 'Error while fetching file %s.' \
                ' Dataset fetching aborted.' % file_name
        if verbose > 1:
            print "HTTP Error:", e, url
        raise
    except urllib2.URLError, e:
        if verbose > 0:
            print 'Error while fetching file %s.' \
                ' Dataset fetching aborted.' % file_name
        if verbose > 1:
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


def _fetch_files(data_dir, files, resume=True, mock=False, verbose=1):
    """Load requested dataset, downloading it if needed or requested.

    This function retrieves files from the hard drive or download them from
    the given urls. Note to developpers: All the files will be first
    downloaded in a sandbox and, if everything goes well, they will be moved
    into the folder of the dataset. This prevents corrupting previously
    downloaded data. In case of a big dataset, do not hesitate to make several
    calls if needed.

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

    mock: boolean, optional
        If true, create empty files if the file cannot be downloaded. Test use
        only.

    verbose: int, optional
        verbosity level (0 means no message).

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
    files_pickle = pickle.dumps(files)
    files_md5 = hashlib.md5(files_pickle).hexdigest()
    temp_dir = os.path.join(data_dir, files_md5)

    # Create destination dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Abortion flag, in case of error
    abort = None

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
        if (abort is None and not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)):

            # We may be in a global read-only repository. If so, we cannot
            # download files.
            if not os.access(data_dir, os.W_OK):
                raise ValueError('Dataset files are missing but dataset'
                                 ' repository is read-only. Contact your data'
                                 ' administrator to solve the problem')

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
                    if not mock or os.path.getsize(dl_file) != 0:
                        _uncompress_file(dl_file, verbose=verbose)
                    else:
                        os.remove(dl_file)
                except Exception as e:
                    abort = str(e)
        if (abort is None and not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)):
            if not mock:
                warnings.warn('An error occured while fetching %s' % file_)
                abort = "Target file cannot be found (%s)" % file_
            else:
                if not os.path.exists(os.path.dirname(temp_target_file)):
                    os.makedirs(os.path.dirname(temp_target_file))
                open(temp_target_file, 'w').close()
        if abort is not None:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise IOError('Fetching aborted: ' + abort)
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

def fetch_craddock_2011_atlas(data_dir=None, url=None, resume=True, verbose=1):
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

    if url is None:
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

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    params = dict(zip(keys, sub_files))
    return Bunch(**params)


def fetch_yeo_2011_atlas(data_dir=None, url=None, resume=True, verbose=1):
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

        - "thin_7", "thick_7": 7-region parcellations,
          fitted to resp. thin and thick template cortex segmentations.

        - "thin_17", "thick_17": 17-region parcellations.

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
    module_path = os.path.dirname(__file__)
    if url is None:
        url = "ftp://surfer.nmr.mgh.harvard.edu/" \
              "pub/data/Yeo_JNeurophysiol11_MNI152.zip"
    opts = {'uncompress': True}

    dataset_name = "yeo_2011"
    keys = ("thin_7", "thick_7",
            "thin_17", "thick_17",
            "colors_7", "colors_17", "anat")
    basenames = (
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_7Networks_ColorLUT.txt",
        "Yeo2011_17Networks_ColorLUT.txt",
        "FSL_MNI152_FreeSurferConformed_1mm.nii.gz")

    filenames = [(os.path.join("Yeo_JNeurophysiol11_MNI152", f), url, opts)
                 for f in basenames]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
            verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    with open(os.path.join(module_path, 'description', 'yeo.rst'))\
            as rst_file:
        fdescr = rst_file.read()

    params = dict([('description', fdescr)] + zip(keys, sub_files))
    return Bunch(**params)


def fetch_icbm152_2009(data_dir=None, url=None, resume=True, verbose=1):
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

    if url is None:
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

    data_dir = _get_dataset_dir('icbm152_2009', data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    params = dict(zip(keys, sub_files))
    return Bunch(**params)


def fetch_smith_2009(data_dir=None, url=None, resume=True,
        verbose=1):
    """Download and load the Smith ICA and BrainMap atlas (dated 2009)

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
        dictionary-like object, contains:
        - 20-dimensional ICA, Resting-FMRI components:
            - all 20 components (rsn20)
            - 10 well-matched maps from these, as shown in PNAS paper (rsn10)

        - 20-dimensional ICA, BrainMap components:
            - all 20 components (bm20)
            - 10 well-matched maps from these, as shown in PNAS paper (bm10)

        - 70-dimensional ICA, Resting-FMRI components (rsn70)

        - 70-dimensional ICA, BrainMap components (bm70)


    References
    ----------

    S.M. Smith, P.T. Fox, K.L. Miller, D.C. Glahn, P.M. Fox, C.E. Mackay, N.
    Filippini, K.E. Watkins, R. Toro, A.R. Laird, and C.F. Beckmann.
    Correspondence of the brain's functional architecture during activation and
    rest. Proc Natl Acad Sci USA (PNAS), 106(31):13040-13045, 2009.

    A.R. Laird, P.M. Fox, S.B. Eickhoff, J.A. Turner, K.L. Ray, D.R. McKay, D.C
    Glahn, C.F. Beckmann, S.M. Smith, and P.T. Fox. Behavioral interpretations
    of intrinsic connectivity networks. Journal of Cognitive Neuroscience, 2011

    Notes
    -----
    For more information about this dataset's structure:
    http://www.fmrib.ox.ac.uk/analysis/brainmap+rsns/
    """

    if url is None:
        url = "http://www.fmrib.ox.ac.uk/analysis/brainmap+rsns/"

    files = [('rsn20.nii.gz', url + 'rsn20.nii.gz', {}),
             ('PNAS_Smith09_rsn10.nii.gz',
                 url + 'PNAS_Smith09_rsn10.nii.gz', {}),
             ('rsn70.nii.gz', url + 'rsn70.nii.gz', {}),
             ('bm20.nii.gz', url + 'bm20.nii.gz', {}),
             ('PNAS_Smith09_bm10.nii.gz',
                 url + 'PNAS_Smith09_bm10.nii.gz', {}),
             ('bm70.nii.gz', url + 'bm70.nii.gz', {}),
             ]

    data_dir = _get_dataset_dir('smith_2009', data_dir=data_dir,
                                verbose=verbose)
    files_ = _fetch_files(data_dir, files, resume=resume,
                          verbose=verbose)

    return Bunch(rsn20=files_[0], rsn10=files_[1], rsn70=files_[2],
            bm20=files_[3], bm10=files_[4], bm70=files_[5])


def fetch_haxby_simple(data_dir=None, url=None, resume=True, verbose=1):
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

    data_dir = _get_dataset_dir('haxby2001_simple', data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    # return the data
    return Bunch(func=files[1], session_target=files[0], mask=files[2],
                 conditions_target=files[3])


def fetch_haxby(data_dir=None, n_subjects=1, fetch_stimuli=False,
                url=None, resume=True, verbose=1):
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

    data_dir = _get_dataset_dir('haxby2001', data_dir=data_dir,
                                verbose=verbose)

    # Dataset files
    if url is None:
        url = 'http://data.pymvpa.org/datasets/haxby2001/'
    md5sums = _fetch_files(data_dir, [('MD5SUMS', url + 'MD5SUMS', {})],
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

    files = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    if n_subjects == 6:
        files.append(None)  # None value because subject 6 has no anat

    kwargs = {}
    if fetch_stimuli:
        files = [(os.path.join('stimuli', 'README'),
                  url + 'stimuli-2010.01.14.tar.gz',
                  {'uncompress': True})]
        readme = _fetch_files(data_dir, files, resume=resume,
                              verbose=verbose)[0]
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
                   verbose=1):
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

    data_dir = _get_dataset_dir('nyu_rest', data_dir=data_dir, verbose=verbose)
    anat_anon = _fetch_files(data_dir, anat_anon, resume=resume,
                             verbose=verbose)
    anat_skull = _fetch_files(data_dir, anat_skull, resume=resume,
                              verbose=verbose)
    func = _fetch_files(data_dir, func, resume=resume,
                        verbose=verbose)

    return Bunch(anat_anon=anat_anon, anat_skull=anat_skull, func=func,
                 session=session)


def fetch_adhd(n_subjects=None, data_dir=None, url=None, resume=True,
               verbose=1):
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

    data_dir = _get_dataset_dir('adhd', data_dir=data_dir, verbose=verbose)
    subjects_funcs = _fetch_files(data_dir, subjects_funcs, resume=resume,
                                  verbose=verbose)
    subjects_confounds = _fetch_files(data_dir, subjects_confounds,
            resume=resume, verbose=verbose)
    phenotypic = _fetch_files(data_dir, phenotypic, resume=resume,
                              verbose=verbose)[0]

    # Load phenotypic data
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',',
                               dtype=None)
    # Keep phenotypic information for selected subjects
    isubs = np.asarray(subs, dtype=int)
    phenotypic = phenotypic[[np.where(phenotypic['Subject'] == i)[0][0]
                             for i in isubs]]

    return Bunch(func=subjects_funcs, confounds=subjects_confounds,
                 phenotypic=phenotypic)


def fetch_msdl_atlas(data_dir=None, url=None, resume=True, verbose=1):
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
        https://team.inria.fr/parietal/files/2015/01/MSDL_rois.zip

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
    url = 'https://team.inria.fr/parietal/files/2015/01/MSDL_rois.zip'
    opts = {'uncompress': True}

    dataset_name = "msdl_atlas"
    files = [(os.path.join('MSDL_rois', 'msdl_rois_labels.csv'), url, opts),
             (os.path.join('MSDL_rois', 'msdl_rois.nii'), url, opts)]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return Bunch(labels=files[0], maps=files[1])


def fetch_harvard_oxford(atlas_name, data_dir=None, symmetric_split=False,
                        resume=True, verbose=1):
    """Load Harvard-Oxford parcellation from FSL if installed or download it.

    This function looks up for Harvard Oxford atlas in the system and load it
    if present. If not, it downloads it and store it in NILEARN_DATA
    directory.

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

    data_dir: string, optional
        Path of data directory. It can be FSL installation directory
        (which is dependent on your installation).

    symmetric_split: bool, optional
        If True, split every symmetric region in left and right parts.
        Effectively doubles the number of regions. Default: False.
        Not implemented for probabilistic atlas (*-prob-* atlases)

    Returns
    =======
    regions: nibabel.Nifti1Image
        regions definition, as a label image.
    """
    atlas_items = ("cort-maxprob-thr0-1mm", "cort-maxprob-thr0-2mm",
                   "cort-maxprob-thr25-1mm", "cort-maxprob-thr25-2mm",
                   "cort-maxprob-thr50-1mm", "cort-maxprob-thr50-2mm",
                   "sub-maxprob-thr0-1mm", "sub-maxprob-thr0-2mm",
                   "sub-maxprob-thr25-1mm", "sub-maxprob-thr25-2mm",
                   "sub-maxprob-thr50-1mm", "sub-maxprob-thr50-2mm",
                   "cort-prob-1mm", "cort-prob-2mm",
                   "sub-prob-1mm", "sub-prob-2mm")
    if atlas_name not in atlas_items:
        raise ValueError("Invalid atlas name: {0}. Please chose an atlas "
                         "among:\n{1}".format(
                             atlas_name, '\n'.join(atlas_items)))

    # Add FSL_DIR as a non-readable data_dir
    data_dirs = []
    fsl_dir = os.getenv('FSL_DIR', os.getenv('FSLDIR'))
    if fsl_dir is not None:
        data_dirs.append(fsl_dir)
    if data_dir is not None:
        data_dirs.append(data_dir)
    data_dir = ':'.join(data_dirs) if len(data_dirs) > 0 else None

    # grab data from internet first
    url = 'https://www.nitrc.org/frs/download.php/7363/HarvardOxford.tgz'
    dataset_name = 'harvard_oxford'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                lookup_defaults=(data_dir is None),
                                verbose=verbose)
    opts = {'uncompress': True}
    atlas_file = os.path.join('HarvardOxford',
                              'HarvardOxford-' + atlas_name + '.nii.gz')
    if atlas_name[0] == 'c':
        label_file = 'HarvardOxford-Cortical.xml'
    else:
        label_file = 'HarvardOxford-Subcortical.xml'

    atlas_img, label_file = _fetch_files(
        data_dir,
        [(atlas_file, url, opts), (label_file, url, opts)],
        resume=resume, verbose=verbose)

    names = {}
    from xml.etree import ElementTree
    names[0] = 'Background'
    for label in ElementTree.parse(label_file).findall('.//label'):
        names[int(label.get('index')) + 1] = label.text
    names = np.asarray(names.values())

    if not symmetric_split:
        return atlas_img, names

    if atlas_name in ("cort-prob-1mm", "cort-prob-2mm",
                      "sub-prob-1mm", "sub-prob-2mm"):
        raise ValueError("Region splitting not supported for probabilistic "
                         "atlases")

    atlas_img = nibabel.load(atlas_img)
    atlas = atlas_img.get_data()

    labels = np.unique(atlas)
    slices = ndimage.find_objects(atlas)
    middle_ind = (atlas.shape[0] - 1) / 2
    crosses_middle = [s.start < middle_ind and s.stop > middle_ind
             for s, _, _ in slices]

    # Split every zone crossing the median plane into two parts.
    # Assumes that the background label is zero.
    half = np.zeros(atlas.shape, dtype=np.bool)
    half[:middle_ind, ...] = True
    new_label = max(labels) + 1
    # Put zeros on the median plane
    atlas[middle_ind, ...] = 0
    for label, crosses in zip(labels[1:], crosses_middle):
        if not crosses:
            continue
        atlas[np.logical_and(atlas == label, half)] = new_label
        new_label += 1

    # Duplicate labels for right and left
    new_names = [names[0]]
    for n in names[1:]:
        new_names.append(n + ', right part')
    for n in names[1:]:
        new_names.append(n + ', left part')

    return nibabel.Nifti1Image(atlas, atlas_img.get_affine()), new_names


def fetch_miyawaki2008(data_dir=None, url=None, resume=True, verbose=1):
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

    data_dir = _get_dataset_dir('miyawaki2008', data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, file_names, resume=resume, verbose=verbose)

    # Return the data
    return Bunch(
        func=files[:32],
        label=files[32:64],
        mask=files[64],
        mask_roi=files[65:])


def fetch_localizer_contrasts(contrasts, n_subjects=None, get_tmaps=False,
                              get_masks=False, get_anats=False,
                              data_dir=None, url=None, resume=True, verbose=1):
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
        or equivalently on can use the original names:
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

    """
    if isinstance(contrasts, basestring):
        raise ValueError('Constrasts should be a list of string, a single '
                         'string was given: "%s"' % contrasts)
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
    allowed_contrasts = contrast_name_wrapper.values()
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
               urllib.quote(base_query % {"types": rql_types,
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
                       urllib.quote(base_query % {"types": '"boolean mask"',
                                                  "label": "mask"},
                                    safe=',()')))
        for subject_id in subject_ids:
            file_path = os.path.join(
                "brainomics_data", subject_id, "boolean_mask_mask.nii.gz")
            file_tarball_url = urls[-1]
            filenames.append((file_path, file_tarball_url, opts))
    # Fetch anats if asked by user
    if get_anats:
        urls.append("%sbrainomics_data_anats.zip?rql=%s&vid=data-zip"
                    % (root_url,
                       urllib.quote(base_query % {"types": '"normalized T1"',
                                                  "label": "anatomy"},
                                    safe=',()')))
        for subject_id in subject_ids:
            file_path = os.path.join(
                "brainomics_data", subject_id,
                "normalized_T1_anat_defaced.nii.gz")
            file_tarball_url = urls[-1]
            filenames.append((file_path, file_tarball_url, opts))
    # Fetch subject characteristics (separated in two files)
    if url is None:
        url_csv = ("%sdataset/cubicwebexport.csv?rql=%s&vid=csvexport"
                   % (root_url, urllib.quote("Any X WHERE X is Subject")))
        url_csv2 = ("%sdataset/cubicwebexport2.csv?rql=%s&vid=csvexport"
                    % (root_url,
                       urllib.quote("Any X,XI,XD WHERE X is QuestionnaireRun, "
                                    "X identifier XI, X datetime "
                                    "XD", safe=',')
                       ))
    else:
        url_csv = "%s/cubicwebexport.csv" % url
        url_csv2 = "%s/cubicwebexport2.csv" % url
    filenames += [("cubicwebexport.csv", url_csv, {}),
                  ("cubicwebexport2.csv", url_csv2, {})]

    # Actual data fetching
    data_dir = _get_dataset_dir('brainomics_localizer', data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, filenames, verbose=verbose)
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
                 ext_vars=csv_data)


def fetch_localizer_calculation_task(n_subjects=None, data_dir=None, url=None,
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

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'cmaps': string list
            Paths to nifti contrast maps

    """
    data = fetch_localizer_contrasts(["calculation (auditory and visual cue)"],
                                     n_subjects=n_subjects,
                                     get_tmaps=False, get_masks=False,
                                     get_anats=False, data_dir=data_dir,
                                     url=url, resume=True, verbose=verbose)
    data.pop('tmaps')
    data.pop('masks')
    data.pop('anats')
    return data


def fetch_oasis_vbm(n_subjects=None, dartel_version=True,
                    data_dir=None, url=None, resume=True, verbose=1):
    """Download and load Oasis "cross-sectional MRI" dataset (416 subjects).

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load. If None is given, all the
        subjects are used.

    dartel_version: boolean,
        Whether or not to use data normalized with DARTEL instead of standard
        SPM8 normalization.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume: bool, optional
        If true, try resuming download if possible

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'gray_matter_maps': string list
            Paths to nifti gray matter density probability maps
        'white_matter_maps' string list
            Paths to nifti gray matter density probability maps
        'ext_vars': np.recarray
            Data from the .csv file with information about selected subjects
        'data_usage_agreement': string
            Path to the .txt file containing the data usage agreement.

    References
    ----------
    [1] http://www.oasis-brains.org/

    [2] Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI
        Data in Young, Middle Aged, Nondemented, and Demented Older Adults.
        Marcus, D. S and al., 2007, Journal of Cognitive Neuroscience.

    Notes
    -----
    In the DARTEL version, original Oasis data [1] have been preprocessed
    with the following steps:
      1. Dimension swapping (technically required for subsequent steps)
      2. Brain Extraction
      3. Segmentation with SPM8
      4. Normalization using DARTEL algorithm
      5. Modulation
      6. Replacement of NaN values with 0 in gray/white matter density maps.
      7. Resampling to reduce shape and make it correspond to the shape of
         the non-DARTEL data (fetched with dartel_version=False).
      8. Replacement of values < 1e-4 with zeros to reduce the file size.

    In the non-DARTEL version, the following steps have been performed instead:
      1. Dimension swapping (technically required for subsequent steps)
      2. Brain Extraction
      3. Segmentation and normalization to a template with SPM8
      4. Modulation
      5. Replacement of NaN values with 0 in gray/white matter density maps.

    An archive containing the gray and white matter density probability maps
    for the 416 available subjects is provided. Gross outliers are removed and
    filtered by this data fetcher (DARTEL: 13 outliers; non-DARTEL: 1 outlier)
    Externals variates (age, gender, estimated intracranial volume,
    years of education, socioeconomic status, dementia score) are provided
    in a CSV file that is a copy of the original Oasis CSV file. The current
    downloader loads the CSV file and keeps only the lines corresponding to
    the subjects that are actually demanded.

    The Open Access Structural Imaging Series (OASIS) is a project
    dedicated to making brain imaging data openly available to the public.
    Using data available through the OASIS project requires agreeing with
    the Data Usage Agreement that can be found at
    http://www.oasis-brains.org/app/template/UsageAgreement.vm

    """
    # check number of subjects
    if n_subjects is None:
        n_subjects = 403 if dartel_version else 415
    if dartel_version:  # DARTEL version has 13 identified outliers
        if n_subjects > 403:
            warnings.warn('Only 403 subjects are available in the '
                          'DARTEL-normalized version of the dataset. '
                          'All of them will be used instead of the wanted %d'
                          % n_subjects)
            n_subjects = 403
    else:  # all subjects except one are available with non-DARTEL version
        if n_subjects > 415:
            warnings.warn('Only 415 subjects are available in the '
                          'non-DARTEL-normalized version of the dataset. '
                          'All of them will be used instead of the wanted %d'
                          % n_subjects)
            n_subjects = 415
    if n_subjects < 1:
        raise ValueError("Incorrect number of subjects (%d)" % n_subjects)

    # pick the archive corresponding to preprocessings type
    if url is None:
        if dartel_version:
            url_images = ('https://www.nitrc.org/frs/download.php/'
                          '6364/archive_dartel.tgz?i_agree=1&download_now=1')
        else:
            url_images = ('https://www.nitrc.org/frs/download.php/'
                          '6359/archive.tgz?i_agree=1&download_now=1')
        # covariates and license are in separate files on NITRC
        url_csv = ('https://www.nitrc.org/frs/download.php/'
                   '6348/oasis_cross-sectional.csv?i_agree=1&download_now=1')
        url_dua = ('https://www.nitrc.org/frs/download.php/'
                   '6349/data_usage_agreement.txt?i_agree=1&download_now=1')
    else:  # local URL used in tests
        url_csv = url + "/oasis_cross-sectional.csv"
        url_dua = url + "/data_usage_agreement.txt"
        if dartel_version:
            url_images = url + "/archive_dartel.tgz"
        else:
            url_images = url + "/archive.tgz"

    opts = {'uncompress': True}

    # missing subjects create shifts in subjects ids
    missing_subjects = [8, 24, 36, 48, 89, 93, 100, 118, 128, 149, 154,
                        171, 172, 175, 187, 194, 196, 215, 219, 225, 242,
                        245, 248, 251, 252, 257, 276, 297, 306, 320, 324,
                        334, 347, 360, 364, 391, 393, 412, 414, 427, 436]

    if dartel_version:
        # DARTEL produces outliers that are hidden by Nilearn's API
        removed_outliers = [27, 57, 66, 83, 122, 157, 222, 269, 282, 287,
                            309, 428]
        missing_subjects = sorted(missing_subjects + removed_outliers)
        file_names_gm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwrc1OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects][:n_subjects]
        file_names_wm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwrc2OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects]
    else:
        # only one gross outlier produced, hidden by Nilearn's API
        removed_outliers = [390]
        missing_subjects = sorted(missing_subjects + removed_outliers)
        file_names_gm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwc1OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects][:n_subjects]
        file_names_wm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwc2OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects]
    file_names_extvars = [("oasis_cross-sectional.csv", url_csv, {})]
    file_names_dua = [("data_usage_agreement.txt", url_dua, {})]
    # restrict to user-specified number of subjects
    file_names_gm = file_names_gm[:n_subjects]
    file_names_wm = file_names_wm[:n_subjects]

    file_names = (file_names_gm + file_names_wm
                  + file_names_extvars + file_names_dua)
    data_dir = _get_dataset_dir('oasis1', data_dir=data_dir, verbose=verbose)
    files = _fetch_files(data_dir, file_names, resume=resume,
                         verbose=verbose)

    # Build Bunch
    gm_maps = files[:n_subjects]
    wm_maps = files[n_subjects:(2 * n_subjects)]
    ext_vars_file = files[-2]
    data_usage_agreement = files[-1]

    # Keep CSV information only for selected subjects
    csv_data = np.recfromcsv(ext_vars_file)
    actual_subjects_ids = ["OAS1"
                           + str.split(os.path.basename(x), "OAS1")[1][:9]
                           for x in gm_maps]
    subject_mask = np.zeros(csv_data.size, dtype=bool)
    for i, subject_id in enumerate(csv_data['id']):
        if subject_id in actual_subjects_ids:
            subject_mask[i] = True
    csv_data = csv_data[subject_mask]

    return Bunch(
        gray_matter_maps=gm_maps,
        white_matter_maps=wm_maps,
        ext_vars=csv_data,
        data_usage_agreement=data_usage_agreement)


def load_mni152_template():
    """Load skullstripped 2mm version of the MNI152 originally distributed
    with FSL

    Returns
    -------
    mni152_template: nibabel object corresponding to the template


    References
    ----------

    VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins and
    BDCG, Unbiased average age-appropriate atlases for pediatric studies,
    NeuroImage, Volume 54, Issue 1, January 2011, ISSN 1053–8119, DOI:
    10.1016/j.neuroimage.2010.07.033

    VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins, Unbiased
    nonlinear average age-appropriate brain templates from birth to adulthood,
    NeuroImage, Volume 47, Supplement 1, July 2009, Page S102 Organization for
    Human Brain Mapping 2009 Annual Meeting, DOI: 10.1016/S1053-8119(09)70884-5

    """
    package_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(package_directory, "data", "avg152T1_brain.nii.gz")

    return nibabel.load(path)


def fetch_abide_pcp(data_dir=None, n_subjects=None, pipeline='cpac',
                    strategy='nofilt_noglobal', derivatives=['func_preproc'],
                    quality_checked=True, url=None, verbose=1, **kwargs):
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

    pipeline: string, optional
        Possible pipelines are "ccs", "cpac", "dparsf" and "niak"

    strategy: string, optional
        Nuisance removal strategy. The first part describe the band filtering
        strategy (either "filt" or "nofilt") and the second part indicate the
        global signal regression strategy ("global" or "noglobal").

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

    # Parameter check
    for derivative in derivatives:
        if not derivative in ['alff', 'degree_binarize', 'degree_weighted',
                'dual_regression', 'eigenvector_binarize',
                'eigenvector_weighted', 'falff', 'func_mask', 'func_mean',
                'func_preproc', 'lfcd', 'reho', 'rois_aal', 'rois_cc200',
                'rois_cc400', 'rois_dosenbach160', 'rois_ez', 'rois_ho',
                'rois_tt', 'vmhc']:
            raise KeyError('%s is not a valid derivative' % derivative)

    # General file: phenotypic information
    data_dir = _get_dataset_dir('ABIDE_pcp', data_dir=data_dir,
                                verbose=verbose)
    if url is None:
        url = ('https://s3.amazonaws.com/fcp-indi/data/Projects/'
               'ABIDE_Initiative')

    if quality_checked:
        kwargs['qc_rater_1'] = 'OK'
        kwargs['qc_anat_rater_2'] = ['OK', 'maybe']
        kwargs['qc_func_rater_2'] = ['OK', 'maybe']
        kwargs['qc_anat_rater_3'] = 'OK'
        kwargs['qc_func_rater_3'] = 'OK'

    # Fetch the phenotypic file and load it
    csv = 'Phenotypic_V1_0b_preprocessed1.csv'
    path_csv = _fetch_files(data_dir, [(csv, url + '/' + csv, {})],
                            verbose=verbose)[0]

    # Note: the phenotypic file contains string that contains comma which mess
    # up numpy array csv loading. This is why I do a pass to remove the last
    # field. This can be
    # done simply with pandas but we don't want such dependency ATM
    # pheno = pandas.read_csv(path_csv).to_records()
    pheno_f = open(path_csv, 'r')
    pheno = ['i' + pheno_f.readline()]
    # This regexp replaces commas between double quotes
    for line in pheno_f:
        pheno.append(re.sub(r',(?=[^"]*"(?:[^"]*"[^"]*")*[^"]*$)', ";", line))
    pheno_f.close()
    pheno = '\n'.join(pheno)
    pheno = StringIO.StringIO(pheno)
    # We enforce empty comments because it is 'sharp' by default
    pheno = np.recfromcsv(pheno, comments=[], case_sensitive=True)

    # First, filter subjects with no filename
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']
    # Apply user defined filters
    filter = _filter_columns(pheno, kwargs)
    pheno = pheno[filter]

    # Go into specific data folder and url
    data_dir = os.path.join(data_dir, pipeline, strategy)
    url = '/'.join([url, 'Outputs', pipeline, strategy])

    # Get the files
    results = {}
    file_ids = pheno['FILE_ID']
    if n_subjects is not None:
        file_ids = file_ids[:n_subjects]
        pheno = pheno[:n_subjects]

    results['phenotypic'] = pheno
    for derivative in derivatives:
        ext = '.1D' if derivative.startswith('rois') else '.nii.gz'
        files = [(file_id + '_' + derivative + ext,
                  '/'.join([url, derivative,
                            file_id + '_' + derivative + ext]),
                  {}) for file_id in file_ids]
        files = _fetch_files(data_dir, files, verbose=verbose)
        # Load derivatives if needed
        if ext == '.1D':
            files = [np.loadtxt(f) for f in files]
        results[derivative] = files

    return Bunch(**results)


def fetch_mixed_gambles(n_subjects=1, data_dir=None, url=None, resume=True,
                        verbose=0, make_Xy=False, smooth=0.):
    """Fetch Jimura "mixed gambles" dataset


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

    resume: bool, optional (default True)
        If true, try resuming download if possible.

    verbose: int, optional (default 0)
        Defines the level of verbosity of the output.

    maks_Xy: bool, optional (default False)
        If true, then the data will transformed into and (X, y) pair, suitable
        for machine learning routines.
        X is a list of n_subjects * 48 Nifti1Image objects, and
        y is an array of shape (n_subjects * 48,).

    smooth: float, or list of 3 floats, optional (default 0.)
        Size of smoothing kernel to apply to the loaded z_maps.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'z_maps': string list
            Paths to realigned gain betamaps (one nifti per subject).
        'X': list of Nifi1Image objects, or None
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

    # fetch files
    if url is None:
        url = ("https://www.nitrc.org/frs/download.php/7229/"
               "jimura_poldrack_2012_zmaps.zip")
    opts = {'uncompress': True}
    files = [("zmaps/sub%03i_zmaps.nii.gz" % (j + 1), url, opts)
             for j in xrange(n_subjects)]
    data_dir = _get_dataset_dir('jimura_poldrack_2012_zmaps',
                                data_dir=data_dir)
    zmap_fnames = _fetch_files(data_dir, files, resume=resume,
                               verbose=verbose)
    data = Bunch(z_maps=zmap_fnames)

    # make data for learning problems, etc.
    if make_Xy:
        if verbose:
            print "Chewing data into matrices X, y..."
        X = []
        y = []
        mask = []
        for zmap_fname in zmap_fnames:
            # load subject data
            img = nibabel.load(zmap_fname)
            this_X = img.get_data()
            affine = img.get_affine()
            finite_mask = np.all(np.isfinite(this_X), axis=-1)
            this_mask = np.logical_and(np.all(this_X != 0, axis=-1),
                                       finite_mask)

            # smooth data ?
            if smooth:
                for i in range(this_X.shape[-1]):
                    this_X[..., i] = ndimage.gaussian_filter(this_X[..., i],
                                                             smooth)
                this_X[np.logical_not(finite_mask)] = np.nan
            this_y = np.array([np.arange(1, 9)] * 6).ravel()

            # gain levels
            if len(this_y) != this_X.shape[-1]:
                raise RuntimeError("%s: Expecting %i volumes, got %i!" % (
                    zmap_fname, len(this_y), this_X.shape[-1]))

            # standardize subject data
            this_X -= this_X.mean(axis=-1)[..., np.newaxis]
            std = this_X.std(axis=-1)
            std[std == 0] = 1
            this_X /= std[..., np.newaxis]

            # commit subject data
            X.append(this_X)
            y.extend(this_y)
            mask.append(this_mask)

        # final sip
        y = np.array(y)
        X = np.concatenate(X, axis=-1)
        mask = np.sum(mask, axis=0) > .5 * len(zmap_fnames)
        mask = np.logical_and(mask, np.all(np.isfinite(X), axis=-1))
        X = X[mask, :].T
        tmp = np.zeros(list(mask.shape) + [len(X)])
        tmp[mask, :] = X.T
        mask_img = nibabel.Nifti1Image(mask.astype(np.int), affine)
        X = nibabel.four_to_three(nibabel.Nifti1Image(tmp, affine))
        if len(X) != y.size:
            raise RuntimeError
        data.update(X=X, y=y, mask_img=mask_img)

    return data
