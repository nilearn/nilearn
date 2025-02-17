"""Private utility functions to the datasets module."""

import collections.abc
import contextlib
import fnmatch
import hashlib
import os
import pickle
import shutil
import tarfile
import time
import urllib
import warnings
import zipfile
from pathlib import Path

import numpy as np
import requests

from nilearn._utils import fill_doc, logger

from .utils import get_data_dirs

_REQUESTS_TIMEOUT = (15.1, 61)
PACKAGE_DIRECTORY = Path(__file__).absolute().parent


ALLOWED_DATA_TYPES = (
    "curvature",
    "sulcal",
    "thickness",
)

ALLOWED_MESH_TYPES = {
    "pial",
    "white_matter",
    "inflated",
    "sphere",
    "flat",
}


def md5_hash(string):
    """Calculate the MD5 hash of a string."""
    m = hashlib.md5()
    m.update(string.encode("utf-8"))
    return m.hexdigest()


def _format_time(t):
    return f"{t / 60.0:4.1f}min" if t > 60 else f" {t:5.1f}s"


def _md5_sum_file(path):
    """Calculate the MD5 sum of a file."""
    with Path(path).open("rb") as f:
        m = hashlib.md5()
        while True:
            data = f.read(8192)
            if data:
                m.update(data)
            else:
                break
    return m.hexdigest()


def read_md5_sum_file(path):
    """Read a MD5 checksum file and returns hashes as a dictionary."""
    with Path(path).open() as f:
        hashes = {}
        while True:
            line = f.readline()
            if not line:
                break
            h, name = line.rstrip().split("  ", 1)
            hashes[name] = h
    return hashes


def _chunk_report_(bytes_so_far, total_size, initial_size, t0):
    """Show downloading percentage.

    Parameters
    ----------
    bytes_so_far : int
        Number of downloaded bytes.

    total_size : int
        Total size of the file (may be 0/None, depending on download method).

    t0 : int
        The time in seconds (as returned by time.time()) at which the
        download was resumed / started.

    initial_size : int
        If resuming, indicate the initial size of the file.
        If not resuming, set to zero.

    """
    if not total_size:
        logger.log(f"\rDownloaded {int(bytes_so_far)} of ? bytes.")

    else:
        # Estimate remaining download time
        total_percent = float(bytes_so_far) / total_size

        current_download_size = bytes_so_far - initial_size
        bytes_remaining = total_size - bytes_so_far
        dt = time.time() - t0
        download_rate = current_download_size / max(1e-8, float(dt))
        # Minimum rate of 0.01 bytes/s, to avoid dividing by zero.
        time_remaining = bytes_remaining / max(0.01, download_rate)

        # Trailing whitespace is to erase extra char when message length varies
        logger.log(
            f"\rDownloaded {bytes_so_far} of {total_size} bytes "
            f"({total_percent * 100:.1f}%%, "
            f"{_format_time(time_remaining)} remaining)",
        )


@fill_doc
def _chunk_read_(
    response,
    local_file,
    chunk_size=8192,
    report_hook=None,
    initial_size=0,
    total_size=None,
    verbose=1,
):
    """Download a file chunk by chunk and show advancement.

    Parameters
    ----------
    response : urllib.response.addinfourl
        Response to the download request in order to get file size.

    local_file : file
        Hard disk file where data should be written.

    chunk_size : int, default=8192
        Size of downloaded chunks.

    report_hook : bool, optional
        Whether or not to show downloading advancement. Default: None

    initial_size : int, default=0
        If resuming, indicate the initial size of the file.

    total_size : int, optional
        Expected final size of download (None means it is unknown).
    %(verbose)s

    Returns
    -------
    data : string
        The downloaded file.

    """
    try:
        if total_size is None:
            total_size = response.headers.get("Content-Length").strip()
        total_size = int(total_size) + initial_size
    except Exception as e:
        logger.log(
            "Warning: total size could not be determined.",
            verbose=verbose,
            msg_level=2,
        )
        logger.log(
            f"Full stack trace: {e}",
            verbose=verbose,
            msg_level=3,
        )
        total_size = None
    bytes_so_far = initial_size

    t0 = time_last_display = time.time()
    for chunk in response.iter_content(chunk_size):
        bytes_so_far += len(chunk)
        time_last_read = time.time()
        if (
            report_hook
            and
            # Refresh report every second or when download is
            # finished.
            (time_last_read > time_last_display + 1.0 or not chunk)
        ):
            _chunk_report_(bytes_so_far, total_size, initial_size, t0)
            time_last_display = time_last_read
        if chunk:
            local_file.write(chunk)
        else:
            break


@fill_doc
def get_dataset_dir(
    dataset_name, data_dir=None, default_paths=None, verbose=1
):
    """Create if necessary and return data directory of given dataset.

    Parameters
    ----------
    dataset_name : string
        The unique name of the dataset.
    %(data_dir)s
    default_paths : list of string, optional
        Default system paths in which the dataset may already have been
        installed by a third party software. They will be checked first.
    %(verbose)s

    Returns
    -------
    data_dir : pathlib.Path
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
            paths.extend(
                [(Path(d), True) for d in str(default_path).split(os.pathsep)]
            )

    paths.extend([(Path(d), False) for d in get_data_dirs(data_dir=data_dir)])

    logger.log(f"Dataset search paths: {paths}", verbose=verbose, msg_level=2)

    # Check if the dataset exists somewhere
    for path, is_pre_dir in paths:
        if not is_pre_dir:
            path = path / dataset_name
        if path.is_symlink():
            # Resolve path
            path = path.resolve()
        if path.exists() and path.is_dir():
            logger.log(
                f"Dataset found in {path}", verbose=verbose, msg_level=1
            )
            return path

    # If not, create a folder in the first writable directory
    errors = []
    for path, is_pre_dir in paths:
        if not is_pre_dir:
            path = path / dataset_name
        if not path.exists():
            try:
                path.mkdir(parents=True)
                _add_readme_to_default_data_locations(
                    data_dir=data_dir,
                    verbose=verbose,
                )

                logger.log(f"Dataset created in {path}", verbose)

                return path
            except Exception as exc:
                short_error_message = getattr(exc, "strerror", str(exc))
                errors.append(f"\n -{path} ({short_error_message})")

    raise OSError(
        "Nilearn tried to store the dataset in the following "
        f"directories, but: {''.join(errors)}"
    )


def _add_readme_to_default_data_locations(data_dir=None, verbose=1):
    for d in get_data_dirs(data_dir=data_dir):
        file = Path(d) / "README.md"
        if file.parent.exists() and not file.exists():
            with file.open("w") as f:
                f.write(
                    """# Nilearn data folder

This directory is used by Nilearn to store datasets
and atlases downloaded from the internet.
It can be safely deleted.
If you delete it, previously downloaded data will be downloaded again."""
                )

            logger.log(f"Added README.md to {d}", verbose=verbose)


# The functions _is_within_directory and _safe_extract were implemented in
# https://github.com/nilearn/nilearn/pull/3391 to address a directory
# traversal vulnerability https://github.com/advisories/GHSA-gw9q-c7gh-j9vm
def _is_within_directory(directory, target):
    abs_directory = Path(directory).resolve().absolute()
    abs_target = Path(target).resolve().absolute()

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == str(abs_directory)


def _safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    path = Path(path)
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)


@fill_doc
def uncompress_file(file_, delete_archive=True, verbose=1):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    file_ : string
        Path of file to be uncompressed.

    delete_archive : bool, default=True
        Whether or not to delete archive once it is uncompressed.
    %(verbose)s

    Notes
    -----
    This handles zip, tar, gzip and bzip files only.

    """
    logger.log(f"Extracting data from {file_}...", verbose=verbose)

    file_ = Path(file_)
    data_dir = file_.parent

    # We first try to see if it is a zip file
    try:
        filename = data_dir / file_.stem
        with file_.open("rb") as fd:
            header = fd.read(4)
        processed = False
        if zipfile.is_zipfile(file_):
            z = zipfile.ZipFile(file_)
            z.extractall(path=data_dir)
            z.close()
            if delete_archive:
                file_.unlink()
            processed = True
        elif file_.suffix == ".gz" or header.startswith(b"\x1f\x8b"):
            import gzip

            if file_.suffix == ".tgz":
                filename = filename.with_suffix(".tar")
            elif not file_.suffix:
                # We rely on the assumption that gzip files have an extension
                shutil.move(file_, f"{file_}.gz")
                file_ = file_.with_suffix(".gz")
            with gzip.open(file_) as gz, filename.open("wb") as out:
                shutil.copyfileobj(gz, out, 8192)
            # If file is .tar.gz, this will be handled in the next case
            if delete_archive:
                file_.unlink()
            file_ = filename
            processed = True
        if file_.is_file() and tarfile.is_tarfile(file_):
            with contextlib.closing(tarfile.open(file_, "r")) as tar:
                _safe_extract(tar, path=data_dir)
            if delete_archive:
                file_.unlink()
            processed = True
        if not processed:
            raise OSError(f"[Uncompress] unknown archive file format: {file_}")

        logger.log(".. done.\n", verbose=verbose)

    except Exception as e:
        logger.log(f"Error uncompressing file: {e}", verbose=verbose)
        raise


def _filter_column(array, col, criteria):
    """Return index array matching criteria.

    Parameters
    ----------
    array : numpy array with columns
        Array in which data will be filtered.

    col : string
        Name of the column.

    criteria : integer (or float), pair of integers, string or list of these
        if integer, select elements in column matching integer
        if a tuple, select elements between the limits given by the tuple
        if a string, select elements that match the string

    """
    # Raise an error if the column does not exist. This is the only way to
    # test it across all possible types (pandas, recarray...)
    try:
        array[col]
    except Exception:
        raise KeyError(f"Filtering criterion {col} does not exist")

    if (
        not isinstance(criteria, str)
        and not isinstance(criteria, bytes)
        and not isinstance(criteria, tuple)
        and isinstance(criteria, collections.abc.Iterable)
    ):
        filter = np.zeros(array.shape[0], dtype=bool)
        for criterion in criteria:
            filter = np.logical_or(
                filter, _filter_column(array, col, criterion)
            )
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
    if isinstance(criteria, (str, bytes)):
        criteria = np.array(criteria).astype(array[col].dtype)

    return array[col] == criteria


def filter_columns(array, filters, combination="and"):
    """Return indices of recarray entries that match criteria.

    Parameters
    ----------
    array : numpy array with columns
        Array in which data will be filtered.

    filters : list of criteria
        See _filter_column.

    combination : string {'and', 'or'}, default='and'
        String describing the combination operator. Possible values are "and"
        and "or".

    """
    if combination == "and":
        fcomb = np.logical_and
        mask = np.ones(array.shape[0], dtype=bool)
    elif combination == "or":
        fcomb = np.logical_or
        mask = np.zeros(array.shape[0], dtype=bool)
    else:
        raise ValueError(f"Combination mode not known: {combination}")

    for column in filters:
        mask = fcomb(mask, _filter_column(array, column, filters[column]))
    return mask


class _NaiveFTPAdapter(requests.adapters.BaseAdapter):
    def send(
        self,
        request,
        timeout=None,
        **kwargs,  # noqa: ARG002
    ):
        with contextlib.suppress(Exception):
            timeout, _ = timeout
        try:
            data = urllib.request.urlopen(request.url, timeout=timeout)
        except Exception as e:
            raise requests.RequestException(e.reason)
        data.release_conn = data.close
        resp = requests.Response()
        resp.url = data.geturl()
        resp.status_code = data.getcode() or 200
        resp.raw = data
        resp.headers = dict(data.info().items())
        return resp

    def close(self):
        pass


@fill_doc
def fetch_single_file(
    url,
    data_dir,
    resume=True,
    overwrite=False,
    md5sum=None,
    username=None,
    password=None,
    verbose=1,
    session=None,
):
    """Load requested file, downloading it if needed or requested.

    Parameters
    ----------
    %(url)s
    %(data_dir)s
    %(resume)s
    overwrite : bool, default=False
        If true and file already exists, delete it.

    md5sum : string, optional
        MD5 sum of the file. Checked if download of the file is required.

    username : string, optional
        Username used for basic HTTP authentication.

    password : string, optional
        Password used for basic HTTP authentication.
    %(verbose)s
    session : requests.Session, optional
        Session to use to send requests.

    Returns
    -------
    files : pahtlib.Path
        Absolute path of downloaded file.

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded files are
    removed.

    """
    if session is None:
        with requests.Session() as sess:
            sess.mount("ftp:", _NaiveFTPAdapter())
            return fetch_single_file(
                url,
                data_dir,
                resume=resume,
                overwrite=overwrite,
                md5sum=md5sum,
                username=username,
                password=password,
                verbose=verbose,
                session=sess,
            )

    # Determine data path
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename using URL
    parse = urllib.parse.urlparse(url)
    file_name = Path(parse.path).name
    if file_name == "":
        file_name = md5_hash(parse.path)

    temp_file_name = f"{file_name}.part"
    full_name = data_dir / file_name
    temp_full_name = data_dir / temp_file_name
    if full_name.exists():
        if overwrite:
            full_name.unlink()
        else:
            return full_name
    if temp_full_name.exists() and overwrite:
        temp_full_name.unlink()
    t0 = time.time()
    initial_size = 0

    try:
        # Download data
        headers = {}
        auth = None
        if username is not None and password is not None:
            if not url.startswith("https"):
                raise ValueError(
                    "Authentication was requested "
                    f"on a non secured URL ({url})."
                    "Request has been blocked for security reasons."
                )
            auth = (username, password)

        displayed_url = url.split("?")[0] if verbose == 1 else url
        logger.log(f"Downloading data from {displayed_url} ...", verbose)

        if resume and temp_full_name.exists():
            # Download has been interrupted, we try to resume it.
            local_file_size = temp_full_name.stat().st_size
            # If the file exists, then only download the remainder
            headers["Range"] = f"bytes={local_file_size}-"
            try:
                req = requests.Request(
                    method="GET", url=url, headers=headers, auth=auth
                )
                prepped = session.prepare_request(req)
                with session.send(
                    prepped, stream=True, timeout=_REQUESTS_TIMEOUT
                ) as resp:
                    resp.raise_for_status()
                    content_range = resp.headers.get("Content-Range")
                    if content_range is None or not content_range.startswith(
                        f"bytes {local_file_size}-"
                    ):
                        raise OSError("Server does not support resuming")
                    initial_size = local_file_size
                    with temp_full_name.open("ab") as fh:
                        _chunk_read_(
                            resp,
                            fh,
                            report_hook=(verbose > 0),
                            initial_size=initial_size,
                            verbose=verbose,
                        )
            except OSError:
                logger.log(
                    "Resuming failed, try to download the whole file.", verbose
                )
                return fetch_single_file(
                    url,
                    data_dir,
                    resume=False,
                    overwrite=overwrite,
                    md5sum=md5sum,
                    username=username,
                    password=password,
                    verbose=verbose,
                    session=session,
                )
        else:
            req = requests.Request(
                method="GET", url=url, headers=headers, auth=auth
            )
            prepped = session.prepare_request(req)
            with session.send(
                prepped, stream=True, timeout=_REQUESTS_TIMEOUT
            ) as resp:
                resp.raise_for_status()
                with temp_full_name.open("wb") as fh:
                    _chunk_read_(
                        resp,
                        fh,
                        report_hook=(verbose > 0),
                        initial_size=initial_size,
                        verbose=verbose,
                    )
        shutil.move(temp_full_name, full_name)
        dt = time.time() - t0

        # Complete the reporting hook
        logger.log(
            f" ...done. ({dt:.0f} seconds, {dt // 60:.0f} min)\n",
            verbose=verbose,
        )
    except requests.RequestException:
        logger.log(
            f"Error while fetching file {file_name}; dataset fetching aborted."
        )
        raise
    if md5sum is not None and _md5_sum_file(full_name) != md5sum:
        raise ValueError(
            f"File {full_name} checksum verification has failed."
            " Dataset fetching aborted."
        )
    return full_name


def get_dataset_descr(ds_name):
    """Return the description of a dataset."""
    try:
        with (PACKAGE_DIRECTORY / "description" / f"{ds_name}.rst").open(
            "rb"
        ) as rst_file:
            descr = rst_file.read()
    except OSError:
        descr = ""

    if not descr:
        warnings.warn("Could not find dataset description.")

    if isinstance(descr, bytes):
        descr = descr.decode("utf-8")

    return descr


def movetree(src, dst):
    """Move entire tree under `src` inside `dst`.

    Creates `dst` if it does not already exist.

    Any existing file is overwritten.

    The difference with `shutil.mv` is that `shutil.mv` moves `src` under `dst`
    if `dst` already exists.
    """
    src = Path(src)

    # Create destination dir if it does not exist
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    errors = []

    for srcfile in src.iterdir():
        dstfile = dst / srcfile.name
        try:
            if srcfile.is_dir() and dstfile.is_dir():
                movetree(srcfile, dstfile)
                srcfile.rmdir()
            else:
                shutil.move(srcfile, dstfile)
        except OSError as why:
            errors.append((srcfile, dstfile, str(why)))
        # catch the Error from the recursive movetree so that we can
        # continue with other files
        except Exception as err:
            errors.extend(err.args[0])
    if errors:
        raise Exception(errors)


@fill_doc
def fetch_files(data_dir, files, resume=True, verbose=1, session=None):
    """Load requested dataset, downloading it if needed or requested.

    This function retrieves files from the hard drive or download them from
    the given urls. Note to developers: All the files will be first
    downloaded in a sandbox and, if everything goes well, they will be moved
    into the folder of the dataset. This prevents corrupting previously
    downloaded data. In case of a big dataset, do not hesitate to make several
    calls if needed.

    Parameters
    ----------
    %(data_dir)s
    files : list of (string, string, dict)
        List of files and their corresponding url with dictionary that contains
        options regarding the files. Eg. (file_path, url, opt). If a file_path
        is not found in data_dir, as in data_dir/file_path the download will
        be immediately canceled and any downloaded files will be deleted.
        Options supported are:
            * 'move' if renaming the file or moving it to a subfolder is needed
            * 'uncompress' to indicate that the file is an archive
            * 'md5sum' to check the md5 sum of the file
            * 'overwrite' if the file should be re-downloaded even if it exists
    %(resume)s
    %(verbose)s
    session : `requests.Session`, optional
        Session to use to send requests.

    Returns
    -------
    files : list of string
        Absolute paths of downloaded files on disk.

    """
    if session is None:
        with requests.Session() as sess:
            sess.mount("ftp:", _NaiveFTPAdapter())
            return fetch_files(
                data_dir,
                files,
                resume=resume,
                verbose=verbose,
                session=sess,
            )
    # There are two working directories here:
    # - data_dir is the destination directory of the dataset
    # - temp_dir is a temporary directory dedicated to this fetching call. All
    #   files that must be downloaded will be in this directory. If a corrupted
    #   file is found, or a file is missing, this working directory will be
    #   deleted.
    files = list(files)
    files_pickle = pickle.dumps([(file_, url) for file_, url, _ in files])
    files_md5 = hashlib.md5(files_pickle).hexdigest()
    data_dir = Path(data_dir)
    temp_dir = data_dir / files_md5

    # Create destination dir
    data_dir.mkdir(parents=True, exist_ok=True)

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
        target_file = data_dir / file_
        # Target file in temp dir
        temp_target_file = temp_dir / file_
        # Whether to keep existing files
        overwrite = opts.get("overwrite", False)
        if abort is None and (
            overwrite
            or (not target_file.exists() and not temp_target_file.exists())
        ):
            # We may be in a global read-only repository. If so, we cannot
            # download files.
            if not os.access(data_dir, os.W_OK):
                raise ValueError(
                    "Dataset files are missing but dataset"
                    " repository is read-only. Contact your data"
                    " administrator to solve the problem"
                )

            temp_dir.mkdir(parents=True, exist_ok=True)
            md5sum = opts.get("md5sum", None)

            dl_file = fetch_single_file(
                url,
                temp_dir,
                resume=resume,
                verbose=verbose,
                md5sum=md5sum,
                username=opts.get("username", None),
                password=opts.get("password", None),
                session=session,
                overwrite=overwrite,
            )
            if "move" in opts:
                # XXX: here, move is supposed to be a dir, it can be a name
                move = temp_dir / opts["move"]
                move_dir = move.parent
                move_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(dl_file, move)
                dl_file = move
            if "uncompress" in opts:
                try:
                    uncompress_file(dl_file, verbose=verbose)
                except Exception as e:
                    abort = str(e)

        if (
            abort is None
            and not target_file.exists()
            and not temp_target_file.exists()
        ):
            warnings.warn(f"An error occurred while fetching {file_}")
            abort = (
                "Dataset has been downloaded but requested file was "
                f"not provided:\nURL: {url}\n"
                f"Target file: {target_file}\nDownloaded: {dl_file}"
            )
        if abort is not None:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise OSError(f"Fetching aborted: {abort}")
        files_.append(str(target_file))
    # If needed, move files from temps directory to final directory.
    if temp_dir.exists():
        # XXX We could only moved the files requested
        # XXX Movetree can go wrong
        movetree(temp_dir, data_dir)
        shutil.rmtree(temp_dir)
    return files_


def tree(path, pattern=None, dictionary=False):
    """Return a directory tree under the form of a dictionary or list.

    Parameters
    ----------
    path : string or pathlib.Path
        Path browsed.

    pattern : string, optional
        Pattern used to filter files (see fnmatch).

    dictionary : boolean, default=False
        If True, the function will return a dict instead of a list.

    """
    path = Path(path)
    files = []
    dirs = {} if dictionary else []

    for file_path in path.iterdir():
        if file_path.is_dir():
            if dictionary:
                dirs[file_path.name] = tree(file_path, pattern, dictionary)
            else:
                dirs.append(
                    (file_path.name, tree(file_path, pattern, dictionary))
                )
        elif pattern is None or fnmatch.fnmatch(file_path.name, pattern):
            files.append(str(file_path))
    files = sorted(files)
    if not dictionary:
        return sorted(dirs) + files
    if len(dirs) == 0:
        return files
    if len(files) > 0:
        dirs["."] = files
    return dirs
