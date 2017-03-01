"""
Downloading NeuroImaging datasets: utility functions
"""
import os
import numpy as np
import base64
import collections
import contextlib
import fnmatch
import hashlib
import shutil
import time
import sys
import tarfile
import warnings
import zipfile

from .._utils.compat import _basestring, cPickle, _urllib, md5_hash


def _format_time(t):
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % (t)


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file.
    """
    with open(path, 'rb') as f:
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
    with open(path, "r") as f:
        hashes = {}
        while True:
            line = f.readline()
            if not line:
                break
            h, name = line.rstrip().split('  ', 1)
            hashes[name] = h
    return hashes


def readlinkabs(link):
    """
    Return an absolute path for the destination
    of a symlink
    """
    path = os.readlink(link)
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(link), path)


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
        sys.stderr.write("\rDownloaded %d of ? bytes." % (bytes_so_far))

    else:
        # Estimate remaining download time
        total_percent = float(bytes_so_far) / total_size

        current_download_size = bytes_so_far - initial_size
        bytes_remaining = total_size - bytes_so_far
        dt = time.time() - t0
        download_rate = current_download_size / max(1e-8, float(dt))
        # Minimum rate of 0.01 bytes/s, to avoid dividing by zero.
        time_remaining = bytes_remaining / max(0.01, download_rate)

        # Trailing whitespace is to erase extra char when message length
        # varies
        sys.stderr.write(
            "\rDownloaded %d of %d bytes (%.1f%%, %s remaining)"
            % (bytes_so_far, total_size, total_percent * 100,
               _format_time(time_remaining)))


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None,
                 initial_size=0, total_size=None, verbose=1):
    """Download a file chunk by chunk and show advancement

    Parameters
    ----------
    response: _urllib.response.addinfourl
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
    try:
        if total_size is None:
            total_size = response.info().get('Content-Length').strip()
        total_size = int(total_size) + initial_size
    except Exception as e:
        if verbose > 2:
            print("Warning: total size could not be determined.")
            if verbose > 3:
                print("Full stack trace: %s" % e)
        total_size = None
    bytes_so_far = initial_size

    t0 = time_last_display = time.time()
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        time_last_read = time.time()
        if (report_hook and
                # Refresh report every half second or when download is
                # finished.
                (time_last_read > time_last_display + 0.5 or not chunk)):
            _chunk_report_(bytes_so_far,
                           total_size, initial_size, t0)
            time_last_display = time_last_read
        if chunk:
            local_file.write(chunk)
        else:
            break

    return


def get_data_dirs(data_dir=None):
    """ Returns the directories in which nilearn looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    paths: list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NILEARN_SHARED_DATA
    4. the user environment variable NILEARN_DATA
    5. nilearn_data in the user home folder
    """
    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(data_dir.split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv('NILEARN_SHARED_DATA')
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv('NILEARN_DATA')
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        paths.append(os.path.expanduser('~/nilearn_data'))
    return paths


def _get_dataset_dir(dataset_name, data_dir=None, default_paths=None,
                     verbose=1):
    """ Create if necessary and returns data directory of given dataset.

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    default_paths: list of string, optional
        Default system paths in which the dataset may already have been
        installed by a third party software. They will be checked first.

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
    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NILEARN_SHARED_DATA
    4. the user environment variable NILEARN_DATA
    5. nilearn_data in the user home folder
    """
    paths = []
    # Search possible data-specific system paths
    if default_paths is not None:
        for default_path in default_paths:
            paths.extend([(d, True) for d in default_path.split(os.pathsep)])

    paths.extend([(d, False) for d in get_data_dirs(data_dir=data_dir)])

    if verbose > 2:
        print('Dataset search paths: %s' % paths)

    # Check if the dataset exists somewhere
    for path, is_pre_dir in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)
        if os.path.islink(path):
            # Resolve path
            path = readlinkabs(path)
        if os.path.exists(path) and os.path.isdir(path):
            if verbose > 1:
                print('\nDataset found in %s\n' % path)
            return path

    # If not, create a folder in the first writeable directory
    errors = []
    for (path, is_pre_dir) in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                if verbose > 0:
                    print('\nDataset created in %s\n' % path)
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
        sys.stderr.write('Extracting data from %s...' % file_)
    data_dir = os.path.dirname(file_)
    # We first try to see if it is a zip file
    try:
        filename, ext = os.path.splitext(file_)
        with open(file_, "rb") as fd:
            header = fd.read(4)
        processed = False
        if zipfile.is_zipfile(file_):
            z = zipfile.ZipFile(file_)
            z.extractall(path=data_dir)
            z.close()
            if delete_archive:
                os.remove(file_)
            file_ = filename
            processed = True
        elif ext == '.gz' or header.startswith(b'\x1f\x8b'):
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
            processed = True
        if os.path.isfile(file_) and tarfile.is_tarfile(file_):
            with contextlib.closing(tarfile.open(file_, "r")) as tar:
                tar.extractall(path=data_dir)
            if delete_archive:
                os.remove(file_)
            processed = True
        if not processed:
            raise IOError(
                    "[Uncompress] unknown archive file format: %s" % file_)

        if verbose > 0:
            sys.stderr.write('.. done.\n')
    except Exception as e:
        if verbose > 0:
            print('Error uncompressing file: %s' % e)
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

    if (not isinstance(criteria, _basestring) and
        not isinstance(criteria, bytes) and
        not isinstance(criteria, tuple) and
            isinstance(criteria, collections.Iterable)):
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

    # Handle strings with different encodings
    if isinstance(criteria, (_basestring, bytes)):
        criteria = np.array(criteria).astype(array[col].dtype)

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
                md5sum=None, username=None, password=None, handlers=[],
                verbose=1):
    """Load requested file, downloading it if needed or requested.

    Parameters
    ----------
    url: string
        Contains the url of the file to be downloaded.

    data_dir: string
        Path of the data directory. Used for data storage in the specified
        location.

    resume: bool, optional
        If true, try to resume partially downloaded files

    overwrite: bool, optional
        If true and file already exists, delete it.

    md5sum: string, optional
        MD5 sum of the file. Checked if download of the file is required

    username: string, optional
        Username used for basic HTTP authentication

    password: string, optional
        Password used for basic HTTP authentication

    handlers: list of BaseHandler, optional
        urllib handlers passed to urllib.request.build_opener. Used by
        advanced users to customize request handling.

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
    parse = _urllib.parse.urlparse(url)
    file_name = os.path.basename(parse.path)
    if file_name == '':
        file_name = md5_hash(parse.path)

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
        url_opener = _urllib.request.build_opener(*handlers)
        request = _urllib.request.Request(url)
        request.add_header('Connection', 'Keep-Alive')
        if username is not None and password is not None:
            if not url.startswith('https'):
                raise ValueError(
                    'Authentication was requested on a non  secured URL (%s).'
                    'Request has been blocked for security reasons.' % url)
            # Note: HTTPBasicAuthHandler is not fitted here because it relies
            # on the fact that the server will return a 401 error with proper
            # www-authentication header, which is not the case of most
            # servers.
            encoded_auth = base64.b64encode(
                (username + ':' + password).encode())
            request.add_header(b'Authorization', b'Basic ' + encoded_auth)
        if verbose > 0:
            displayed_url = url.split('?')[0] if verbose == 1 else url
            print('Downloading data from %s ...' % displayed_url)
        if resume and os.path.exists(temp_full_name):
            # Download has been interrupted, we try to resume it.
            local_file_size = os.path.getsize(temp_full_name)
            # If the file exists, then only download the remainder
            request.add_header("Range", "bytes=%s-" % (local_file_size))
            try:
                data = url_opener.open(request)
                content_range = data.info().get('Content-Range')
                if (content_range is None or not content_range.startswith(
                        'bytes %s-' % local_file_size)):
                    raise IOError('Server does not support resuming')
            except Exception:
                # A wide number of errors can be raised here. HTTPError,
                # URLError... I prefer to catch them all and rerun without
                # resuming.
                if verbose > 0:
                    print('Resuming failed, try to download the whole file.')
                return _fetch_file(
                    url, data_dir, resume=False, overwrite=overwrite,
                    md5sum=md5sum, username=username, password=password,
                    handlers=handlers, verbose=verbose)
            local_file = open(temp_full_name, "ab")
            initial_size = local_file_size
        else:
            data = url_opener.open(request)
            local_file = open(temp_full_name, "wb")
        _chunk_read_(data, local_file, report_hook=(verbose > 0),
                     initial_size=initial_size, verbose=verbose)
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        shutil.move(temp_full_name, full_name)
        dt = time.time() - t0
        if verbose > 0:
            # Complete the reporting hook
            sys.stderr.write(' ...done. ({0:.0f} seconds, {1:.0f} min)\n'
                             .format(dt, dt // 60))
    except (_urllib.error.HTTPError, _urllib.error.URLError) as e:
        if 'Error while fetching' not in str(e):
            # For some odd reason, the error message gets doubled up
            #   (possibly from the re-raise), so only add extra info
            #   if it's not already there.
            e.reason = ("%s| Error while fetching file %s; "
                          "dataset fetching aborted." % (
                            str(e.reason), file_name))
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


def _get_dataset_descr(ds_name):
    module_path = os.path.dirname(os.path.abspath(__file__))

    fname = ds_name

    try:
        with open(os.path.join(module_path, 'description', fname + '.rst'),
                  'rb') as rst_file:
            descr = rst_file.read()
    except IOError:
        descr = ''

    if descr == '':
        print("Warning: Could not find dataset description.")

    return descr


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
    data_dir: string
        Path of the data directory. Used for data storage in a specified
        location.

    files: list of (string, string, dict)
        List of files and their corresponding url with dictionary that contains
        options regarding the files. Eg. (file_path, url, opt). If a file_path
        is not found in data_dir, as in data_dir/file_path the download will
        be immediately cancelled and any downloaded files will be deleted.
        Options supported are:
            * 'move' if renaming the file or moving it to a subfolder is needed
            * 'uncompress' to indicate that the file is an archive
            * 'md5sum' to check the md5 sum of the file
            * 'overwrite' if the file should be re-downloaded even if it exists

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
    files = list(files)
    files_pickle = cPickle.dumps([(file_, url) for file_, url, _ in files])
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
        # Whether to keep existing files
        overwrite = opts.get('overwrite', False)
        if (abort is None and (overwrite or (not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)))):

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
                                  verbose=verbose, md5sum=md5sum,
                                  username=opts.get('username', None),
                                  password=opts.get('password', None),
                                  handlers=opts.get('handlers', []),
                                  overwrite=overwrite)
            if 'move' in opts:
                # XXX: here, move is supposed to be a dir, it can be a name
                move = os.path.join(temp_dir, opts['move'])
                move_dir = os.path.dirname(move)
                if not os.path.exists(move_dir):
                    os.makedirs(move_dir)
                shutil.move(dl_file, move)
                dl_file = move
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
                abort = ("Dataset has been downloaded but requested file was "
                         "not provided:\nURL: %s\n"
                         "Target file: %s\nDownloaded: %s" %
                         (url, target_file, dl_file))
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
        # XXX We could only moved the files requested
        # XXX Movetree can go wrong
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
