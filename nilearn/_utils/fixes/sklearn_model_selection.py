"""
The :mod:`sklearn.model_selection._split` module includes classes and
functions to split the data based on a preset strategy.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>,
#         Olivier Girsel <olivier.grisel@ensta.org>
#         Raghav R V <rvraghav93@gmail.com>
# License: BSD 3 clause


from __future__ import print_function
from __future__ import division

import warnings
from collections import Iterable
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.utils.multiclass import type_of_target
from sklearn.externals.six import with_metaclass
from sklearn.utils.fixes import signature
from sklearn.base import _pprint
from sklearn.cross_validation import KFold, StratifiedKFold


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
