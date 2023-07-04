"""Misc utilities for the library.

Authors: Bertrand Thirion, Matthew Brett, Ana Luisa Pinho, 2020
"""

import csv
from warnings import warn

import numpy as np
import pandas as pd
import scipy.linalg as spl
from scipy.stats import norm

from nilearn._utils.helpers import stringify_path


def _check_list_length_match(list_1, list_2, var_name_1, var_name_2):
    """Check length match of two given lists to raise error if necessary."""
    if len(list_1) != len(list_2):
        raise ValueError(
            "len(%s) %d does not match len(%s) %d"
            % (str(var_name_1), len(list_1), str(var_name_2), len(list_2))
        )


def _read_events_table(table):
    """Accept the path to en event.tsv file \
    and loads it as a Pandas Dataframe.

    Raises an error if loading fails.

    Parameters
    ----------
    table : string
        Accepts the path to an events file.

    Returns
    -------
    loaded : pandas.Dataframe object
        Pandas Dataframe with e events data.

    """
    try:
        # kept for historical reasons, a lot of tests use csv with index column
        loaded = pd.read_csv(table, index_col=0)
    except:  # noqa: E722
        raise ValueError(f"table path {table} could not be loaded")
    if loaded.empty:
        try:
            loaded = pd.read_csv(table, sep="\t")
        except:  # noqa: E722
            raise ValueError(f"table path {table} could not be loaded")
    return loaded


def _check_and_load_tables(tables_, var_name):
    """Check tables can be loaded in DataFrame to raise error if necessary."""
    tables = []
    for table_idx, table in enumerate(tables_):
        table = stringify_path(table)
        if isinstance(table, str):
            loaded = _read_events_table(table)
            tables.append(loaded)
        elif isinstance(table, pd.DataFrame):
            tables.append(table)
        elif isinstance(table, np.ndarray):
            pass
        else:
            raise TypeError(
                "%s can only be a pandas DataFrames or a"
                "string. A %s was provided at idx %d"
                % (var_name, type(table), table_idx)
            )
    return tables


def _check_events_file_uses_tab_separators(events_files):
    """Raise a ValueError if provided list of text based data files \
    (.csv, .tsv, etc) do not enforce \
    the :term:`BIDS` convention of using Tabs as separators.

    Only scans their first row.
    Does nothing if:
        - If the separator used is :term:`BIDS` compliant.
        - Paths are invalid.
        - File(s) are not text files.

    Does not flag comma-separated-values-files for compatibility reasons;
    this may change in future as commas are not :term:`BIDS` compliant.

    Parameters
    ----------
    events_files : str, List/Tuple[str]
        A single file's path or a collection of filepaths.
        Files are expected to be text files.
        Non-text files will raise ValueError.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If value separators are not Tabs (or commas)

    """
    valid_separators = [",", "\t"]
    if not isinstance(events_files, (list, tuple)):
        events_files = [events_files]
    for events_file_ in events_files:
        try:
            with open(events_file_) as events_file_obj:
                events_file_sample = events_file_obj.readline()
            """
            The following errors are not being handled here,
            as they are handled elsewhere in the calling code.
            Handling them here will beak the calling code,
            and refactoring that is not straightforward.
            """
        except TypeError:  # events is Pandas dataframe.
            pass
        except UnicodeDecodeError:  # py3:if binary file
            raise ValueError(
                "The file does not seem to be a valid unicode text file."
            )
        except OSError:  # if invalid filepath.
            pass
        else:
            try:
                csv.Sniffer().sniff(
                    sample=events_file_sample,
                    delimiters=valid_separators,
                )
            except csv.Error:
                raise ValueError(
                    "The values in the events file "
                    "are not separated by tabs; "
                    "please enforce BIDS conventions",
                    events_file_,
                )


def _check_run_tables(run_imgs, tables_, tables_name):
    """Check fMRI runs and corresponding tables to raise error if necessary."""
    if isinstance(tables_, (str, pd.DataFrame, np.ndarray)):
        tables_ = [tables_]
    _check_list_length_match(run_imgs, tables_, "run_imgs", tables_name)
    tables_ = _check_and_load_tables(tables_, tables_name)
    return tables_


def z_score(pvalue, one_minus_pvalue=None):
    """Return the z-score(s) corresponding to certain p-value(s) and, \
    optionally, one_minus_pvalue(s) provided as inputs.

    Parameters
    ----------
    pvalue : float or 1-d array shape=(n_pvalues,)
        P-values computed using the survival function.

    one_minus_pvalue : float or 1-d array shape=(n_one_minus_pvalues,), \
        optional
        It shall take the value returned
        by /nilearn/glm/contrasts.py::one_minus_pvalue
        which computes the p_value using the cumulative distribution function,
        with n_one_minus_pvalues = n_pvalues.

    Returns
    -------
    z_scores : 1-d array shape=(n_z_scores,), with n_z_scores = n_pvalues

    """
    pvalue = np.clip(pvalue, 1.0e-300, 1.0 - 1.0e-16)
    z_scores_sf = norm.isf(pvalue)

    if one_minus_pvalue is not None:
        one_minus_pvalue = np.clip(one_minus_pvalue, 1.0e-300, 1.0 - 1.0e-16)
        z_scores_cdf = norm.ppf(one_minus_pvalue)
        z_scores = np.empty(pvalue.size)
        use_cdf = z_scores_sf < 0
        use_sf = np.logical_not(use_cdf)
        z_scores[np.atleast_1d(use_cdf)] = z_scores_cdf[use_cdf]
        z_scores[np.atleast_1d(use_sf)] = z_scores_sf[use_sf]
    else:
        z_scores = z_scores_sf
    return z_scores


def multiple_fast_inverse(a):
    """Compute the inverse of a set of arrays.

    Parameters
    ----------
    a : array_like of shape (n_samples, n_dim, n_dim)
        Set of square matrices to be inverted. A is changed in place.

    Returns
    -------
    a : ndarray
       Yielding the inverse of the inputs.

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
        raise ValueError("a must have shape (n_samples, n_dim, n_dim)")
    from scipy.linalg.lapack import get_lapack_funcs

    a1, n = a[0], a.shape[0]
    getrf, getri, getri_lwork = get_lapack_funcs(
        ("getrf", "getri", "getri_lwork"), (a1,)
    )
    for i in range(n):
        if (
            getrf.module_name[:7] == "clapack"
            and getri.module_name[:7] != "clapack"
        ):
            # ATLAS 3.2.1 has getrf but not getri.
            lu, piv, info = getrf(
                np.transpose(a[i]), rowmajor=0, overwrite_a=True
            )
            a[i] = np.transpose(lu)
        else:
            a[i], piv, info = getrf(a[i], overwrite_a=True)
        if info == 0:
            if getri.module_name[:7] == "flapack":
                lwork, _ = getri_lwork(a1.shape[0])
                # XXX: the following line fixes curious SEGFAULT when
                # benchmarking 500x500 matrix inverse. This seems to
                # be a bug in LAPACK ?getri routine because if lwork is
                # minimal (when using lwork[0] instead of lwork[1]) then
                # all tests pass. Further investigation is required if
                # more such SEGFAULTs occur.
                lwork = int(1.01 * lwork.real)
                a[i], _ = getri(a[i], piv, lwork=lwork, overwrite_lu=1)
            else:  # clapack
                a[i], _ = getri(a[i], piv, overwrite_lu=1)
        else:
            raise ValueError("Matrix LU decomposition failed")
    return a


def multiple_mahalanobis(effect, covariance):
    """Return the squared Mahalanobis distance for a given set of samples.

    Parameters
    ----------
    effect : array of shape (n_features, n_samples)
        Each column represents a vector to be evaluated.

    covariance : array of shape (n_features, n_features, n_samples)
        Corresponding covariance models stacked along the last axis.

    Returns
    -------
    sqd : array of shape (n_samples,)
         The squared distances (one per sample).

    """
    # check size
    if effect.ndim == 1:
        effect = effect[:, np.newaxis]
    if covariance.ndim == 2:
        covariance = covariance[:, :, np.newaxis]
    if effect.shape[0] != covariance.shape[0]:
        raise ValueError("Inconsistent shape for effect and covariance")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Inconsistent shape for covariance")

    # transpose and make contuguous for the sake of speed
    Xt, Kt = np.ascontiguousarray(effect.T), np.ascontiguousarray(covariance.T)

    # compute the inverse of the covariances
    Kt = multiple_fast_inverse(Kt)

    # derive the squared Mahalanobis distances
    sqd = np.sum(np.sum(Xt[:, :, np.newaxis] * Xt[:, np.newaxis] * Kt, 1), 1)
    return sqd


def full_rank(X, cmax=1e15):
    """Compute the condition number of X and if it is larger than cmax, \
    returns a matrix with a condition number smaller than cmax.

    Parameters
    ----------
    X : array of shape (nrows, ncols)
        Input array.

    cmax : float, optional
        Tolerance for condition number.
        Default=1e15.

    Returns
    -------
    X : array of shape (nrows, ncols)
        Output array.

    cond : float,
        Actual condition number.

    """
    U, s, V = spl.svd(X, full_matrices=False)
    smax, smin = s.max(), s.min()
    cond = smax / smin
    if cond < cmax:
        return X, cond

    warn("Matrix is singular at working precision, regularizing...")
    lda = (smax - cmax * smin) / (cmax - 1)
    X = np.dot(U, np.dot(np.diag(s + lda), V))
    return X, cmax


def positive_reciprocal(X):
    """Return element-wise reciprocal of array, setting `X`>=0 to 0.

    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    rX : array
       Array of same shape as `X`, dtype float, with values set to
       1/X where X > 0, 0 otherwise.

    """
    X = np.asarray(X)
    return np.where(X <= 0, 0, 1.0 / X)
