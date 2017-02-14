# sklearn 0.18
"""
The :mod:`sklearn.model_selection._split` module includes classes and
functions to split the data based on a preset strategy.
"""
from __future__ import print_function
from __future__ import division

import warnings
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from collections import Iterable
from numpy import bincount

from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.utils.multiclass import type_of_target
from sklearn.externals.six import with_metaclass
from sklearn.base import _pprint

from .sklearn_validation import indexable
from .funcsigs import signature


class BaseCrossValidator(with_metaclass(ABCMeta)):
    """Base class for all cross-validators

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def __init__(self):
        # We need this for the build_repr to work properly in py2.7
        # see #6304
        pass

    def split(self, X, y=None, labels=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, of length n_samples
            The target variable for supervised learning problems.

        labels : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, labels = indexable(X, y, labels)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, labels):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, labels=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, labels)
        """
        for test_index in self._iter_test_indices(X, y, labels):
            test_mask = np.zeros(_num_samples(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, labels=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, labels=None):
        """Returns the number of splitting iterations in the cross-validator"""

    def __repr__(self):
        return _build_repr(self)


class _BaseKFold(with_metaclass(ABCMeta, BaseCrossValidator)):
    """Base class for KFold and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_folds, shuffle, random_state):
        if not isinstance(n_folds, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_folds, type(n_folds)))
        n_folds = int(n_folds)

        if n_folds <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_folds=2 or more,"
                " got n_folds={0}.".format(n_folds))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, labels=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        labels : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, labels = indexable(X, y, labels)
        n_samples = _num_samples(X)
        if self.n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds n_folds={0} greater"
                 " than the number of samples: {1}.").format(self.n_folds,
                                                             n_samples))

        for train, test in super(_BaseKFold, self).split(X, y, labels):
            yield train, test

    def get_n_splits(self, X=None, y=None, labels=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        labels : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_folds


class KFold(_BaseKFold):
    """K-Folds cross-validator

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_folds : int, default=3
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Examples
    --------
    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_folds=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)  # doctest: +NORMALIZE_WHITESPACE
    KFold(n_folds=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in kf.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]

    Notes
    -----
    The first ``n_samples % n_folds`` folds have size
    ``n_samples // n_folds + 1``, other folds have size
    ``n_samples // n_folds``, where ``n_samples`` is the number of samples.

    See also
    --------
    StratifiedKFold
        Takes label information into account to avoid building folds with
        imbalanced class distributions (for binary or multiclass
        classification tasks).

    LabelKFold: K-fold iterator variant with non-overlapping labels.
    """

    def __init__(self, n_folds=3, shuffle=False,
                 random_state=None):
        super(KFold, self).__init__(n_folds, shuffle, random_state)

    def _iter_test_indices(self, X, y=None, labels=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_folds = self.n_folds
        fold_sizes = (n_samples // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n_samples % n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class StratifiedKFold(_BaseKFold):
    """Stratified K-Folds cross-validator

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_folds : int, default=3
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.

    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Examples
    --------
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_folds=2)
    >>> skf.get_n_splits(X, y)
    2
    >>> print(skf)  # doctest: +NORMALIZE_WHITESPACE
    StratifiedKFold(n_folds=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in skf.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]

    Notes
    -----
    All the folds have size ``trunc(n_samples / n_folds)``, the last one has
    the complementary.

    """

    def __init__(self, n_folds=3, shuffle=False, random_state=None):
        super(StratifiedKFold, self).__init__(n_folds, shuffle, random_state)

    def _make_test_folds(self, X, y=None, labels=None):
        if self.shuffle:
            rng = check_random_state(self.random_state)
        else:
            rng = self.random_state
        y = np.asarray(y)
        n_samples = y.shape[0]
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts = bincount(y_inversed)
        min_labels = np.min(y_counts)
        if np.all(self.n_folds > y_counts):
            raise ValueError("All the n_labels for individual classes"
                             " are less than %d folds."
                             % (self.n_folds))
        if self.n_folds > min_labels:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of labels for any class cannot"
                           " be less than n_folds=%d."
                           % (min_labels, self.n_folds)), Warning)

        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each class so as to respect the balance of
        # classes
        # NOTE: Passing the data corresponding to ith class say X[y==class_i]
        # will break when the data is not 100% stratifiable for all classes.
        # So we pass np.zeroes(max(c, n_folds)) as data to the KFold
        per_cls_cvs = [
            KFold(self.n_folds, shuffle=self.shuffle,
                  random_state=rng).split(np.zeros(max(count, self.n_folds)))
            for count in y_counts]

        test_folds = np.zeros(n_samples, dtype=np.int)
        for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
            for cls, (_, test_split) in zip(unique_y, per_cls_splits):
                cls_test_folds = test_folds[y == cls]
                # the test split can be too big because we used
                # KFold(...).split(X[:max(c, n_folds)]) when data is not 100%
                # stratifiable for all the classes
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(cls_test_folds)]
                cls_test_folds[test_split] = test_fold_indices
                test_folds[y == cls] = cls_test_folds

        return test_folds

    def _iter_test_masks(self, X, y=None, labels=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_folds):
            yield test_folds == i

    def split(self, X, y, labels=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        labels : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super(StratifiedKFold, self).split(X, y, labels)


class _CVIterableWrapper(BaseCrossValidator):
    """Wrapper class for old style cv objects and iterables."""
    def __init__(self, cv):
        self.cv = cv

    def get_n_splits(self, X=None, y=None, labels=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        labels : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.cv)  # Both iterables and old-cv objects support len

    def split(self, X=None, y=None, labels=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        labels : object
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        for train, test in self.cv:
            yield train, test


def check_cv(cv=3, y=None, classifier=False):
    """Input checker utility for building a cross-validator

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    y : array-like, optional
        The target variable for supervised learning problems.

    classifier : boolean, optional, default False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    """
    if cv is None:
        cv = 3

    if isinstance(cv, numbers.Integral):
        if (classifier and (y is not None) and
                (type_of_target(y) in ('binary', 'multiclass'))):
            return StratifiedKFold(cv)
        else:
            return KFold(cv)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             "or an iterable. Got %s." % cv)
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))
