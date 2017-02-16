""" Misc utilities for the library

Authors: Bertrand Thirion, Matthew Brett, 2015
"""
import sys
import scipy.linalg as spl
import numpy as np
from scipy.stats import norm
from warnings import warn
import pandas as pd
import os
import glob

py3 = sys.version_info[0] >= 3


def _check_list_length_match(list_1, list_2, var_name_1, var_name_2):
    """Check length match of two given lists to raise error if necessary"""
    if len(list_1) != len(list_2):
        raise ValueError(
            'len(%s) %d does not match len(%s) %d'
            % (str(var_name_1), len(list_1), str(var_name_2), len(list_2)))


def _check_and_load_tables(tables_, var_name):
    """Check tables can be loaded in DataFrame to raise error if necessary"""
    tables = []
    for table_idx, table in enumerate(tables_):
        if isinstance(table, _basestring):
            try:
                loaded = pd.read_csv(table, index_col=0)
            except:
                raise ValueError('table path %s could not be loaded' % table)
            tables.append(loaded)
        elif isinstance(table, pd.DataFrame):
            tables.append(table)
        else:
            raise TypeError('%s can only be a pandas DataFrames or a'
                            'string. A %s was provided at idx %d' %
                            (var_name, type(table), table_idx))
    return tables


def _check_run_tables(run_imgs, tables_, tables_name):
    """Check fMRI runs and corresponding tables to raise error if necessary"""
    if isinstance(tables_, (_basestring, pd.DataFrame)):
        tables_ = [tables_]
    _check_list_length_match(run_imgs, tables_, 'run_imgs', tables_name)
    tables_ = _check_and_load_tables(tables_, tables_name)
    return tables_


def z_score(pvalue):
    """ Return the z-score corresponding to a given p-value.
    """
    pvalue = np.minimum(np.maximum(pvalue, 1.e-300), 1. - 1.e-16)
    return norm.isf(pvalue)


def multiple_fast_inv(a):
    """Compute the inverse of a set of arrays.

    Parameters
    ----------
    a: array_like of shape (n_samples, n_dim, n_dim)
        Set of square matrices to be inverted. A is changed in place.

    Returns
    -------
    a: ndarray
       yielding the inverse of the inputs

    Raises
    ------
    LinAlgError :
        If `a` is singular.
    ValueError :
        If `a` is not square, or not 2-dimensional.

    Notes
    -----
    This function is borrowed from scipy.linalg.inv,
    but with some customizations for speed-up.
    """
    if a.shape[1] != a.shape[2]:
        raise ValueError('a must have shape (n_samples, n_dim, n_dim)')
    from scipy.linalg import calc_lwork
    from scipy.linalg.lapack import get_lapack_funcs
    a1, n = a[0], a.shape[0]
    getrf, getri = get_lapack_funcs(('getrf', 'getri'), (a1,))
    for i in range(n):
        if (getrf.module_name[:7] == 'clapack'
            and getri.module_name[:7] != 'clapack'):
            # ATLAS 3.2.1 has getrf but not getri.
            lu, piv, info = getrf(np.transpose(a[i]), rowmajor=0,
                                  overwrite_a=True)
            a[i] = np.transpose(lu)
        else:
            a[i], piv, info = getrf(a[i], overwrite_a=True)
        if info == 0:
            if getri.module_name[:7] == 'flapack':
                lwork = calc_lwork.getri(getri.prefix, a1.shape[0])
                lwork = lwork[1]
                # XXX: the following line fixes curious SEGFAULT when
                # benchmarking 500x500 matrix inverse. This seems to
                # be a bug in LAPACK ?getri routine because if lwork is
                # minimal (when using lwork[0] instead of lwork[1]) then
                # all tests pass. Further investigation is required if
                # more such SEGFAULTs occur.
                lwork = int(1.01 * lwork)
                a[i], _ = getri(a[i], piv, lwork=lwork, overwrite_lu=1)
            else:  # clapack
                a[i], _ = getri(a[i], piv, overwrite_lu=1)
        else:
            raise ValueError('Matrix LU decomposition failed')
    return a


def multiple_mahalanobis(effect, covariance):
    """Returns the squared Mahalanobis distance for a given set of samples

    Parameters
    ----------
    effect: array of shape (n_features, n_samples),
        Each column represents a vector to be evaluated

    covariance: array of shape (n_features, n_features, n_samples),
        Corresponding covariance models stacked along the last axis

    Returns
    -------
    sqd: array of shape (n_samples,)
         the squared distances (one per sample)
    """
    # check size
    if effect.ndim == 1:
        effect = effect[:, np.newaxis]
    if covariance.ndim == 2:
        covariance = covariance[:, :, np.newaxis]
    if effect.shape[0] != covariance.shape[0]:
        raise ValueError('Inconsistant shape for effect and covariance')
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError('Inconsistant shape for covariance')

    # transpose and make contuguous for the sake of speed
    Xt, Kt = np.ascontiguousarray(effect.T), np.ascontiguousarray(covariance.T)

    # compute the inverse of the covariances
    Kt = multiple_fast_inv(Kt)

    # derive the squared Mahalanobis distances
    sqd = np.sum(np.sum(Xt[:, :, np.newaxis] * Xt[:, np.newaxis] * Kt, 1), 1)
    return sqd


def full_rank(X, cmax=1e15):
    """ Computes the condition number of X and if it is larger than cmax,
    returns a matrix with a condition number smaller than cmax.

    Parameters
    ----------
    X : array of shape (nrows, ncols)
        input array

    cmax : float, optional (default:1.e15),
        tolerance for condition number

    Returns
    -------
    X : array of shape (nrows, ncols)
        output array

    cond : float,
        actual condition number
    """
    U, s, V = spl.svd(X, full_matrices=False)
    smax, smin = s.max(), s.min()
    cond = smax / smin
    if cond < cmax:
        return X, cond

    warn('Matrix is singular at working precision, regularizing...')
    lda = (smax - cmax * smin) / (cmax - 1)
    X = np.dot(U, np.dot(np.diag(s + lda), V))
    return X, cmax


def pos_recipr(X):
    """ Return element-wise reciprocal of array, setting `X`>=0 to 0

    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    rX : array
       array of same shape as `X`, dtype np.float, with values set to
       1/X where X > 0, 0 otherwise
    """
    X = np.asarray(X)
    return np.where(X <= 0, 0, 1. / X)


_basestring = str if py3 else basestring


# UTILITIES FOR THE BIDS STANDARD
def get_bids_files(main_path, file_tag='*', file_type='*', sub_label='*',
                   modality_folder='*', filters=[], sub_folder=True):
    """Search for files in a BIDS dataset following given constraints.

    This utility function allows to filter files in the BIDS dataset by
    any of the fields contained in the file names. Moreover it allows to search
    for specific types of files or particular tags.

    The provided filters have to correspond to a file name field, so
    any file not containing the field will be ignored. For example the filter
    ('sub', '01') would return all files corresponding to the first
    subject that specifically contain in the file name "sub-01". If more
    filters are given then we constraint the possible files names accordingly.

    Notice that to search in the derivatives folder, it has to be given as
    part of the main_path. This is useful since the current convention gives
    exactly the same inner structure to derivatives than to the main BIDS
    dataset folder, so we can search it in the same way.

    Parameters
    ----------
    main_path: str
        Directory of the BIDS dataset

    file_tag: str accepted by glob, optional (default: '*')
        The final tag of the desired files. For example 'bold' if one is
        interested in the files related to the neuroimages.

    file_type: str accepted by glob, optional (default: '*')
        The type of the desired files. For example to be able to request only
        'nii' or 'json' files for the 'bold' tag.

    sub_label: str accepted by glob, optional (default: '*')
        Such a common filter is given as a direct option since it applies also
        at the level of directories. the label is what follows the 'sub' field
        in the BIDS convention as 'sub-label'.

    modality_folder: str accepted by glob, optional (default: '*')
        Inside the subject and optional session folders a final level of
        folders is expected in the BIDS convention that groups files according
        to different neuroimaging modalities and any other additions of the
        dataset provider. For example the 'func' and 'anat' standard folders.
        If given as the empty string '', files will be searched inside the
        sub-label/ses-label directories.

    filters: list of tuples (str, str), optional (default: [])
        Filters are of the form (field, label). Only one filter per field
        allowed. A file that does not match a filter will be discarded.
        Filter examples would be (ses, 01), (variant, smooth) and
        (task, localizer).

    sub_folder: boolean, optional (default: True)
        Determines if the files searched are at the level of
        subject/session folders or just below the dataset main folder.
        Setting this option to False with other default values would return
        all the files below the main directory, ignoring files in subject
        or derivatives folders.

    Returns
    -------
    files: list of str
        list of file paths found.

    """
    if sub_folder:
        files = os.path.join(main_path, 'sub-*', 'ses-*')
        if glob.glob(files):
            files = os.path.join(main_path, 'sub-%s' % sub_label, 'ses-*',
                                 modality_folder, 'sub-%s*_%s.%s' %
                                 (sub_label, file_tag, file_type))
        else:
            files = os.path.join(main_path, 'sub-%s' % sub_label,
                                 modality_folder, 'sub-%s*_%s.%s' %
                                 (sub_label, file_tag, file_type))
    else:
        files = os.path.join(main_path, '*%s.%s' % (file_tag, file_type))

    files = glob.glob(files)
    files.sort()
    if filters:
        files = [parse_bids_filename(file_) for file_ in files]
        for key, value in filters:
            files = [file_ for file_ in files if (key in file_ and
                                                  file_[key] == value)]
        return [ref_file['file_path'] for ref_file in files]

    return files


def parse_bids_filename(img_path):
    """Returns dictionary with parsed information from file path

    Parameters
    ----------
    img_path: str

    Returns
    -------
    reference: dict
        returns a dictionary with all key-value pairs in the file name
        parsed and other useful fields like 'file_path', 'file_basename',
        'file_tag', 'file_type' and 'file_fields'.

        The 'file_tag' field refers to the last part of the file under the
        BIDS convention that is of the form *_tag.type. Contrary to the rest
        of the file name it is not a key-value pair. This notion should be
        revised in the case we are handling derivatives since so far the
        convention will keep the tag prepended to any fields added in the
        case of preprocessed files that also end with another tag. This parser
        will consider any tag in the middle of the file name as a key with no
        value and will be included in the 'file_fields' key.
    """
    reference = {}
    reference['file_path'] = img_path
    reference['file_basename'] = os.path.basename(img_path)
    parts = reference['file_basename'].split('_')
    tag, type_ = parts[-1].split('.', 1)
    reference['file_tag'] = tag
    reference['file_type'] = type_
    reference['file_fields'] = []
    for part in parts[:-1]:
        field = part.split('-')[0]
        reference['file_fields'].append(field)
        # In derivatives is not clear if the source file name will
        # be parsed as a field with no value.
        if len(part.split('-')) > 1:
            value = part.split('-')[1]
            reference[field] = value
        else:
            reference[field] = None
    return reference
