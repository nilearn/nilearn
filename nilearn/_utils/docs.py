# -*- coding: utf-8 -*-
"""Functions related to the documentation.

docdict contains the standard documentation entries
used accross Nilearn.

source: Eric Larson and MNE-python team.
https://github.com/mne-tools/mne-python/blob/main/mne/utils/docs.py
"""

import sys

###################################
# Standard documentation entries
#
docdict = dict()

# Verbose
verbose = """
verbose : int, optional
    Verbosity level (0 means no message).
    Default={}."""
docdict['verbose'] = verbose.format(1)
docdict['verbose0'] = verbose.format(0)

# Resume
docdict['resume'] = """
resume : bool, optional
    Whether to resumed download of a partly-downloaded file.
    Default=True."""

# Data_dir
docdict['data_dir'] = """
data_dir : string, optional
    Path where data should be downloaded. By default,
    files are downloaded in home directory."""

# URL
docdict['url'] = """
url : string, optional
    URL of file to download.
    Override download URL. Used for test only (or if you
    setup a mirror of the data).
    Default=None."""

# Smoothing_fwhm
docdict['smoothing_fwhm'] = """
smoothing_fwhm : float, optional.
    If smoothing_fwhm is not None, it gives the size in millimeters of the
    spatial smoothing to apply to the signal.
    Default=None."""

# Standardize
standardize = """
standardize : bool, optional.
    If standardize is True, the data are centered and normed:
    their variance is put to 1 in the time dimension.
    Default={}."""
docdict['standardize'] = standardize.format('True')
docdict['standardize_false'] = standardize.format('False')

# Target_affine
docdict['target_affine'] = """
target_affine: numpy.ndarray, optional.
    If specified, the image is resampled corresponding to this new affine.
    target_affine can be a 3x3 or a 4x4 matrix.
    Default=None."""

# Target_shape
docdict['target_shape'] = """
target_shape: tuple or list, optional.
    If specified, the image will be resized to match this new shape.
    len(target_shape) must be equal to 3.

    .. note::
        If `target_shape` is specified, a `target_affine` of shape
        (4, 4) must also be given.

    Default=None."""

# Low_pass
docdict['low_pass'] = """
low_pass: float, optional
    Low cutoff frequency in Hertz.
    Default=None."""

# High pass
docdict['high_pass'] = """
high_pass: float, optional
    High cutoff frequency in Hertz.
    Default=None."""

# t_r
docdict['t_r'] = """
t_r: float, optional
    Repetition time, in second (sampling period). Set to None if not.
    Default=None."""

# Memory
docdict['memory'] = """
memory : instance of joblib.Memory or str
    Used to cache the masking process.
    By default, no caching is done. If a str is given, it is the
    path to the caching directory."""

# Memory_level
memory_level = """
memory_level: int, optional.
    Rough estimator of the amount of memory used by caching. Higher value
    means more memory for caching.
    Default={}."""
docdict['memory_level'] = memory_level.format(0)
docdict['memory_level1'] = memory_level.format(1)

# n_jobs
n_jobs = """
n_jobs : int, optional.
    The number of CPUs to use to do the computation. -1 means 'all CPUs'.
    Default={}."""
docdict['n_jobs'] = n_jobs.format("1")
docdict['n_jobs_all'] = n_jobs.format("-1")

# fsaverage options
docdict['fsaverage_options'] = """

        - 'fsaverage3': the low-resolution fsaverage3 mesh (642 nodes)
        - 'fsaverage4': the low-resolution fsaverage4 mesh (2562 nodes)
        - 'fsaverage5': the low-resolution fsaverage5 mesh (10242 nodes)
        - 'fsaverage5_sphere': the low-resolution fsaverage5 spheres

            .. deprecated:: 0.8.0
                This option has been deprecated and will be removed in v0.9.0.
                fsaverage5 sphere coordinates can now be accessed through
                attributes sphere_{left, right} using mesh='fsaverage5'

        - 'fsaverage6': the medium-resolution fsaverage6 mesh (40962 nodes)
        - 'fsaverage7': same as 'fsaverage'
        - 'fsaverage': the high-resolution fsaverage mesh (163842 nodes)

            .. note::
                The high-resolution fsaverage will result in more computation
                time and memory usage

"""

# Classifiers
docdict['classifier_options'] = """

        - `svc`: `Linear support vector classifier <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ with L2 penalty.
            .. code-block:: python

                svc = LinearSVC(penalty='l2', max_iter=1e4)

        - `svc_l2`: `Linear support vector classifier <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ with L2 penalty.
            .. note::
                Same as option `svc`.

        - `svc_l1`: `Linear support vector classifier <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ with L1 penalty.
            .. code-block:: python

                svc_l1 = LinearSVC(penalty='l1', dual=False, max_iter=1e4)

        - `logistic`: `Logistic regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ with L2 penalty.
            .. code-block:: python

                logistic = LogisticRegression(penalty='l2',solver='liblinear')

        - `logistic_l1`: `Logistic regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ with L1 penalty.
            .. code-block:: python

                logistic_l1 = LogisticRegression(penalty='l1', solver='liblinear')

        - `logistic_l2`: `Logistic regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_ with L2 penalty
            .. note::
                Same as option `logistic`.

        - `ridge_classifier`: `Ridge classifier <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html>`_.
            .. code-block:: python

                ridge_classifier = RidgeClassifierCV()

        - `dummy_classifier`: `Dummy classifier <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>`_ with stratified strategy.
            .. code-block:: python

                dummy = DummyClassifier(strategy='stratified', random_state=0)

"""

docdict['regressor_options'] = """

        - `ridge`: `Ridge regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html>`_.
            .. code-block:: python

                ridge = RidgeCV()

        - `ridge_regressor`: `Ridge regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html>`_.
            .. note::
                Same option as `ridge`.

        - `svr`: `Support vector regression <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_.
            .. code-block:: python

                svr = SVR(kernel='linear', max_iter=1e4)

        - `dummy_regressor`: `Dummy regressor <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html>`_.
            .. code-block:: python

                dummy = DummyRegressor(strategy='mean')

"""

docdict_indented = {}


def _indentcount_lines(lines):
    """Minimum indent for all lines in line list

    >>> lines = [' one', '  two', '   three']
    >>> _indentcount_lines(lines)
    1
    >>> lines = []
    >>> _indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> _indentcount_lines(lines)
    1
    >>> _indentcount_lines(['    '])
    0

    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.

    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))
    return f
