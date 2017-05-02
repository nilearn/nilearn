"""
Download statistical maps available on Neurovault (http://neurovault.org).
"""
# Author: Jerome Dockes
# License: simplified BSD

import os
import warnings
import traceback
from copy import copy, deepcopy
import shutil
import re
import json
from glob import glob
from tempfile import mkdtemp
from collections import Container
try:
    # python3
    from urllib.parse import urljoin, urlencode
    from urllib.request import build_opener, Request
    from urllib.error import URLError
except ImportError:
    # python2
    from urlparse import urljoin
    from urllib import urlencode
    from urllib2 import build_opener, Request, URLError

import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction import DictVectorizer

from .._utils.compat import _basestring
from .utils import _fetch_file, _get_dataset_dir, _get_dataset_descr


_NEUROVAULT_BASE_URL = 'http://neurovault.org/api/'
_NEUROVAULT_COLLECTIONS_URL = urljoin(_NEUROVAULT_BASE_URL, 'collections/')
_NEUROVAULT_IMAGES_URL = urljoin(_NEUROVAULT_BASE_URL, 'images/')
_NEUROSYNTH_FETCH_WORDS_URL = 'http://neurosynth.org/api/v2/decode/'

_COL_FILTERS_AVAILABLE_ON_SERVER = ('DOI', 'name', 'owner', 'id')
_IM_FILTERS_AVAILABLE_ON_SERVER = tuple()

_DEFAULT_BATCH_SIZE = 100
_DEFAULT_MAX_IMAGES = 100

# if _MAX_CONSECUTIVE_FAILS downloads fail in a row, we consider there is a
# problem(e.g. no internet connection, or the Neurovault server is down), and
# we abort the fetching.
_MAX_CONSECUTIVE_FAILS = 100

# if _MAX_FAILS_IN_COLLECTION images fail to be downloaded from the same
# collection, we consider this collection is garbage and we move on to the
# next collection.
_MAX_FAILS_IN_COLLECTION = 30

_DEBUG = 3
_INFO = 2
_WARNING = 1
_ERROR = 0


# Helpers for filtering images and collections.

class _SpecialValue(object):
    """Base class for special values used to filter terms.

    Derived classes should override ``__eq__`` in order to create
    objects that can be used for comparisons to particular sets of
    values in filters.

    """
    def __eq__(self, other):
        raise NotImplementedError('Use a derived class for _SpecialValue')

    def __req__(self, other):
        return self.__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __rne__(self, other):
        return self.__ne__(other)

    def __repr__(self):
        if hasattr(self, 'repr_arg_'):
            return '{0}({1!r})'.format(self.__class__.__name__, self.repr_arg_)
        return '{0}()'.format(self.__class__.__name__)


class IsNull(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class will always be equal to, and only to,
    any null value of any type (by null we mean for which bool
    returns False).

    See Also
    --------
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import IsNull
    >>> null = IsNull()
    >>> null == 0
    True
    >>> null == ''
    True
    >>> null == None
    True
    >>> null == 'a'
    False

    """
    def __eq__(self, other):
        return not bool(other)


class NotNull(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class will always be equal to, and only to,
    any non-zero value of any type (by non-zero we mean for which bool
    returns True).

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotNull
    >>> not_null = NotNull()
    >>> not_null == 0
    False
    >>> not_null == ''
    False
    >>> not_null == None
    False
    >>> not_null == 'a'
    True

    """
    def __eq__(self, other):
        return bool(other)


class NotEqual(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with `NotEqual(obj)`. It
    will always be equal to, and only to, any value for which
    ``obj == value`` is ``False``.

    Parameters
    ----------
    negated : object
        The object from which a candidate should be different in order
        to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotEqual
    >>> not_0 = NotEqual(0)
    >>> not_0 == 0
    False
    >>> not_0 == '0'
    True

    """
    def __init__(self, negated):
        self.negated_ = negated
        self.repr_arg_ = self.negated_

    def __eq__(self, other):
        return not self.negated_ == other


class _OrderComp(_SpecialValue):
    """Base class for special values based on order comparisons."""
    def __init__(self, bound):
        self.bound_ = bound
        self._cast = type(bound)
        self.repr_arg_ = self.bound_

    def __eq__(self, other):
        try:
            return self._eq_impl(self._cast(other))
        except (TypeError, ValueError):
            return False


class GreaterOrEqual(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `GreaterOrEqual(obj)`. It
    will always be equal to, and only to, any value for which
    ``obj <= value`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be superior or equal in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import GreaterOrEqual
    >>> nonnegative = GreaterOrEqual(0.)
    >>> nonnegative == -.1
    False
    >>> nonnegative == 0
    True
    >>> nonnegative == .1
    True

    """
    def _eq_impl(self, other):
        return self.bound_ <= other


class GreaterThan(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `GreaterThan(obj)`. It
    will always be equal to, and only to, any value for which
    ``obj < value`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be strictly superior in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import GreaterThan
    >>> positive = GreaterThan(0.)
    >>> positive == 0.
    False
    >>> positive == 1.
    True
    >>> positive == -1.
    False

    """
    def _eq_impl(self, other):
        return self.bound_ < other


class LessOrEqual(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `LessOrEqual(obj)`. It
    will always be equal to, and only to, any value for which
    ``value <= obj`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be inferior or equal in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import LessOrEqual
    >>> nonpositive = LessOrEqual(0.)
    >>> nonpositive == -1.
    True
    >>> nonpositive == 0.
    True
    >>> nonpositive == 1.
    False

    """
    def _eq_impl(self, other):
        return other <= self.bound_


class LessThan(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `LessThan(obj)`. It
    will always be equal to, and only to, any value for which
    ``value < obj`` is ``True``.

    Parameters
    ----------
    bound : object
        The object to which a candidate should be strictly inferior in
        order to pass through the filter.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import LessThan
    >>> negative = LessThan(0.)
    >>> negative == -1.
    True
    >>> negative == 0.
    False
    >>> negative == 1.
    False

    """
    def _eq_impl(self, other):
        return other < self.bound_


class IsIn(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `IsIn(*accepted)`. It will always be equal to, and only to, any
    value for which ``value in accepted`` is ``True``.

    Parameters
    ----------
    accepted : container
        A value will pass through the filter if it is present in
        `accepted`.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import IsIn
    >>> vowels = IsIn('a', 'e', 'i', 'o', 'u', 'y')
    >>> 'a' == vowels
    True
    >>> vowels == 'b'
    False

    """
    def __init__(self, *accepted):
        self.accepted_ = accepted

    def __eq__(self, other):
        return other in self.accepted_

    def __repr__(self):
        return '{0}{1!r}'.format(
            self.__class__.__name__, self.accepted_)


class NotIn(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `NotIn(*rejected)`. It will always be equal to, and only to, any
    value for which ``value in rejected`` is ``False``.

    Parameters
    ----------
    rejected : container
        A value will pass through the filter if it is absent from
        `rejected`.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotIn
    >>> consonants = NotIn('a', 'e', 'i', 'o', 'u', 'y')
    >>> 'b' == consonants
    True
    >>> consonants == 'a'
    False

    """
    def __init__(self, *rejected):
        self.rejected_ = rejected

    def __eq__(self, other):
        return other not in self.rejected_

    def __repr__(self):
        return '{0}{1!r}'.format(
            self.__class__.__name__, self.rejected_)


class Contains(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `Contains(*must_be_contained)`. It will always be equal to, and
    only to, any value for which ``item in value`` is ``True`` for
    every item in ``must_be_contained``.

    Parameters
    ----------
    must_be_contained : container
        A value will pass through the filter if it contains all the
        items in must_be_contained.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import Contains
    >>> contains = Contains('house', 'face')
    >>> 'face vs house' == contains
    True
    >>> 'smiling face vs frowning face' == contains
    False

    """
    def __init__(self, *must_be_contained):
        self.must_be_contained_ = must_be_contained

    def __eq__(self, other):
        if not isinstance(other, Container):
            return False
        for item in self.must_be_contained_:
            if item not in other:
                return False
        return True

    def __repr__(self):
        return '{0}{1!r}'.format(
            self.__class__.__name__, self.must_be_contained_)


class NotContains(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `NotContains(*must_not_be_contained)`. It will always be equal
    to, and only to, any value for which ``item in value`` is
    ``False`` for every item in ``must_not_be_contained``.

    Parameters
    ----------
    must_not_be_contained : container
        A value will pass through the filter if it does not contain
        any of the items in must_not_be_contained.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import NotContains
    >>> no_garbage = NotContains('bad', 'test')
    >>> no_garbage == 'test image'
    False
    >>> no_garbage == 'good image'
    True

    """
    def __init__(self, *must_not_be_contained):
        self.must_not_be_contained_ = must_not_be_contained

    def __eq__(self, other):
        if not isinstance(other, Container):
            return False
        for item in self.must_not_be_contained_:
            if item in other:
                return False
        return True

    def __repr__(self):
        return '{0}{1!r}'.format(
            self.__class__.__name__, self.must_not_be_contained_)


class Pattern(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with

    `Pattern(pattern[, flags])`. It will always be equal to, and only
    to, any value for which ``re.match(pattern, value, flags)`` is
    ``True``.

    Parameters
    ----------
    pattern : str
        The pattern to try to match to candidates.

    flags : int, optional (default=0)
        Value for ``re.match`` `flags` parameter,
        e.g. ``re.IGNORECASE``. The default (0), is the default value
        used by ``re.match``.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains.

    Documentation for standard library ``re`` module.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import Pattern
    >>> poker = Pattern(r'[0-9akqj]{5}$')
    >>> 'ak05q' == poker
    True
    >>> 'ak05e' == poker
    False

    """
    def __init__(self, pattern, flags=0):
        # Don't use re.compile because compiled patterns
        # can't be deepcopied.
        self.pattern_ = pattern
        self.flags_ = flags

    def __eq__(self, other):
        if not isinstance(other, _basestring) or re.match(
                self.pattern_, other, self.flags_) is None:
            return False
        return True

    def __repr__(self):
        return '{0}(pattern={1!r}, flags={2})'.format(
            self.__class__.__name__, self.pattern_, self.flags_)


def _empty_filter(arg):
    """Place holder for a filter which always returns True.

    This is the default ``image_filter`` and ``collection_filter``
    argument for ``fetch_neurovault``.

    """
    return True


class ResultFilter(object):

    """Easily create callable (local) filters for ``fetch_neurovault``.

    Constructed from a mapping of key-value pairs (optional) and a
    callable filter (also optional), instances of this class are meant
    to be used as ``image_filter`` or ``collection_filter`` parameters
    for ``fetch_neurovault``.

    Such filters can be combined using their methods ``AND``, ``OR``,
    ``XOR``, and ``NOT``, with the usual semantics.

    Key-value pairs can be added by treating a ``ResultFilter`` as a
    dictionary: after evaluating ``res_filter[key] = value``, only
    metadata such that ``metadata[key] == value`` can pass through the
    filter.

    Parameters
    ----------

    query_terms : dict, optional (default=None)
        A ``metadata`` dictionary will be blocked by the filter if it
        does not respect ``metadata[key] == value`` for all
        ``key``, ``value`` pairs in `query_terms`. If ``None``, the
        empty dictionary is used.

    callable_filter : callable, optional (default=_empty_filter)
        A ``metadata`` dictionary will be blocked by the filter if
        `callable_filter` does not return ``True`` for ``metadata``.

    As an alternative to the `query_terms` dictionary parameter,
    key, value pairs can be passed as keyword arguments.

    Attributes
    ----------
    query_terms_ : dict
        In order to pass through the filter, metadata must verify
        ``metadata[key] == value`` for each ``key``, ``value`` pair in
        `query_terms_`.

    callable_filters_ : list of callables
        In addition to ``(key, value)`` pairs, we can use this
        attribute to specify more elaborate requirements. Called with
        a dict representing metadata for an image or collection, each
        element of this list returns ``True`` if the metadata should
        pass through the filter and ``False`` otherwise.

    A dict of metadata will only pass through the filter if it
    satisfies all the `query_terms` AND all the elements of
    `callable_filters_`.

    See Also
    --------
    nilearn.datasets.neurovault.IsNull,
    nilearn.datasets.neurovault.NotNull,
    nilearn.datasets.neurovault.NotEqual,
    nilearn.datasets.neurovault.GreaterOrEqual,
    nilearn.datasets.neurovault.GreaterThan,
    nilearn.datasets.neurovault.LessOrEqual,
    nilearn.datasets.neurovault.LessThan,
    nilearn.datasets.neurovault.IsIn,
    nilearn.datasets.neurovault.NotIn,
    nilearn.datasets.neurovault.Contains,
    nilearn.datasets.neurovault.NotContains,
    nilearn.datasets.neurovault.Pattern.

    Examples
    --------
    >>> from nilearn.datasets.neurovault import ResultFilter
    >>> filt = ResultFilter(a=0).AND(ResultFilter(b=1).OR(ResultFilter(b=2)))
    >>> filt({'a': 0, 'b': 1})
    True
    >>> filt({'a': 0, 'b': 0})
    False

    """

    def __init__(self, query_terms=None,
                 callable_filter=_empty_filter, **kwargs):
        if query_terms is None:
            query_terms = {}
        query_terms = dict(query_terms, **kwargs)
        self.query_terms_ = query_terms
        self.callable_filters_ = [callable_filter]

    def __call__(self, candidate):
        """Return True if candidate satisfies the requirements.

        Parameters
        ----------
        candidate : dict
            A dictionary representing metadata for a file or a
            collection, to be filtered.

        Returns
        -------
        bool
            ``True`` if `candidate` passes through the filter and ``False``
            otherwise.

        """
        for key, value in self.query_terms_.items():
            if not (value == candidate.get(key)):
                return False
        for callable_filter in self.callable_filters_:
            if not callable_filter(candidate):
                return False
        return True

    def OR(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) or filt2(r))
        return new_filter

    def AND(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) and filt2(r))
        return new_filter

    def XOR(self, other_filter):
        filt1, filt2 = deepcopy(self), deepcopy(other_filter)
        new_filter = ResultFilter(
            callable_filter=lambda r: filt1(r) != filt2(r))
        return new_filter

    def NOT(self):
        filt = deepcopy(self)
        new_filter = ResultFilter(
            callable_filter=lambda r: not filt(r))
        return new_filter

    def __getitem__(self, item):
        """Get item from query_terms_"""
        return self.query_terms_[item]

    def __setitem__(self, item, value):
        """Set item in query_terms_"""
        self.query_terms_[item] = value

    def __delitem__(self, item):
        """Remove item from query_terms_"""
        if item in self.query_terms_:
            del self.query_terms_[item]

    def add_filter(self, callable_filter):
        """Add a function to the callable_filters_.

        After a call add_filter(additional_filt), in addition to all
        the previous requirements, a candidate must also verify
        additional_filt(candidate) in order to pass through the
        filter.

        """
        self.callable_filters_.append(callable_filter)

    def __str__(self):
        return self.__class__.__name__


# Utilities for composing queries and interacting with
# neurovault and neurosynth

class _TemporaryDirectory(object):
    """Context manager that provides a temporary directory

    A temporary directory is created on __enter__
    and removed on __exit__ .

    Attributes
    ----------
    temp_dir_ : str or None
        location of temporary directory or None if not created.

    """

    def __init__(self):
        self.temp_dir_ = None

    def __enter__(self):
        self.temp_dir_ = mkdtemp()
        return self.temp_dir_

    def __exit__(self, *args):
        if self.temp_dir_ is None:
            return
        shutil.rmtree(self.temp_dir_)
        self.temp_dir_ = None


def _print_if(message, level, threshold_level,
              with_traceback=False):
    """Print a message if its importance is above a threshold.

    Parameters
    ----------
    message : str
        the message to print if `level` is strictly above
        `threshold_level`.

    level : int
        importance of the message.

    threshold_level : int
        the message is printed if `level` is strictly above
        `threshold_level`.

    with_traceback : bool, optional (default=False)
        if `message` is printed, also print the last traceback.

    """
    if level > threshold_level:
        return
    print(message)
    if with_traceback:
        traceback.print_exc()


def _append_filters_to_query(query, filters):
    """Encode dict or sequence of key-value pairs into a URL query string

    Parameters
    ----------
    query : str
        URL to which the filters should be appended

    filters : dict or sequence of pairs
        Filters to append to the URL.

    Returns
    -------
    str
        The query with filters appended to it.

    Notes
    -----
    If one of the `filters` keys is 'id', we get the url that points
    directly to that id,
    e.g. 'http://neurovault.org/api/collections/40', and the other
    filters are ignored.

    """
    if not filters:
        return query
    if 'id' in filters:
        return urljoin(query, str(filters['id']))
    new_query = urljoin(
        query, '?{0}'.format(urlencode(filters)))
    return new_query


def _get_encoding(resp):
    """Get the encoding of an HTTP response.

    Parameters
    ----------
    resp : http.client.HTTPResponse
        Response whose encoding we want to find out.

    Returns
    -------
    str
        str representing the encoding, e.g. 'utf-8'.

    Raises
    ------
    ValueError
        If the response does not specify an encoding.

    """
    try:
        charset = resp.headers.get_content_charset()
        if charset is not None:
            return charset
    except AttributeError:
        pass
    content_type = resp.headers.get('Content-Type', '')
    match = re.search(r'charset=\b(.+)\b', content_type)
    if match is None:
        raise ValueError(
            'HTTP response encoding not found; headers: {0}'.format(
                resp.headers))
    return match.group(1)


def _get_batch(query, prefix_msg='', timeout=10., verbose=3):
    """Given an URL, get the HTTP response and transform it to python dict.

    The URL is used to send an HTTP GET request and the response is
    transformed into a dictionary.

    Parameters
    ----------
    query : str
        The URL from which to get data.

    prefix_msg : str, optional (default='')
        Prefix for all log messages.

    timeout : float
        Timeout in seconds.

    verbose : int, optional (default=3)
        an integer in [0, 1, 2, 3] to control the verbosity level.

    Returns
    -------
    batch : dict
        Python dict representing the response's content.

    Raises
    ------
    urllib.error.URLError
        If there was a problem opening the URL.

    ValueError
        If the response could not be decoded, or did not contain
        either 'id' (single result), or 'results' and 'count' (actual
        batch).

    Notes
    -----
    urllib.error.HTTPError is a subclass of URLError.

    """
    request = Request(query)
    request.add_header('Connection', 'Keep-Alive')
    opener = build_opener()
    _print_if('{0}getting new batch: {1}'.format(
        prefix_msg, query), _DEBUG, verbose)
    try:
        resp = opener.open(request, timeout=timeout)

    except Exception:
        _print_if('Could not download batch from {0}'.format(query),
                  _ERROR, verbose, with_traceback=True)
        raise
    try:
        encoding = _get_encoding(resp)
        content = resp.read()
        batch = json.loads(content.decode(encoding))
    except(URLError, ValueError):
        _print_if('Could not decypher batch from {0}'.format(query),
                  _ERROR, verbose, with_traceback=True)
        raise
    finally:
        resp.close()
    if 'id' in batch:
        batch = {'count': 1, 'results': [batch]}
    for key in ['results', 'count']:
        if batch.get(key) is None:
            msg = ('Could not find required key "{0}" '
                   'in batch retrieved from {1}'.format(key, query))
            _print_if(msg, _ERROR, verbose)
            raise ValueError(msg)

    return batch


def _scroll_server_results(url, local_filter=_empty_filter,
                           query_terms=None, max_results=None,
                           batch_size=None, prefix_msg='', verbose=3):
    """Download list of metadata from Neurovault.

    Parameters
    ----------
    url : str
        The base url (without the filters) from which to get data.

    local_filter : callable, optional (default=_empty_filter)
        Used to filter the results based on their metadata:
        must return True if the result is to be kept and False otherwise.
        Is called with the dict containing the metadata as sole argument.

    query_terms : dict, sequence of pairs or None, optional (default=None)
        Key-value pairs to add to the base url in order to form query.
        If ``None``, nothing is added to the url.

    max_results: int or None, optional (default=None)
        Maximum number of results to fetch; if ``None``, all available data
        that matches the query is fetched.

    batch_size: int or None, optional (default=None)
        Neurovault returns the metadata for hits corresponding to a query
        in batches. batch_size is used to choose the (maximum) number of
        elements in a batch. If None, ``_DEFAULT_BATCH_SIZE`` is used.

    prefix_msg: str, optional (default='')
        Prefix for all log messages.

    verbose : int, optional (default=3)
        an integer in [0, 1, 2, 3] to control the verbosity level.

    Yields
    ------
    result : dict
        A result in the retrieved batch.

    None
        Once for each batch that could not be downloaded or decoded,
        to indicate a failure.

    """
    query = _append_filters_to_query(url, query_terms)
    if batch_size is None:
        batch_size = _DEFAULT_BATCH_SIZE
    query = '{0}{1}limit={2}&offset={{0}}'.format(
        query, ('&' if '?' in query else '?'), batch_size)
    downloaded = 0
    n_available = None
    while(max_results is None or downloaded < max_results):
        new_query = query.format(downloaded)
        try:
            batch = _get_batch(new_query, prefix_msg, verbose=verbose)
        except Exception:
            yield None
            batch = None
        if batch is not None:
            batch_size = len(batch['results'])
            downloaded += batch_size
            _print_if('{0}batch size: {1}'.format(prefix_msg, batch_size),
                      _DEBUG, verbose)
            if n_available is None:
                n_available = batch['count']
                max_results = (n_available if max_results is None
                               else min(max_results, n_available))
            for result in batch['results']:
                if local_filter(result):
                    yield result


def _yield_from_url_list(url_list, verbose=3):
    """Get metadata coming from an explicit list of URLs.

    This is different from ``_scroll_server_results``, which is used
    to get all the metadata that matches certain filters.

    Parameters
    ----------
    url_list : Container of str
        URLs from which to get data

    verbose : int, optional (default=3)
        an integer in [0, 1, 2, 3] to control the verbosity level.

    Yields
    ------
    content : dict
        The metadata from one URL.

    None
        Once for each URL that resulted in an error, to signify failure.

    """
    for url in url_list:
        try:
            batch = _get_batch(url, verbose=verbose)
        except Exception:
            yield None
            batch = None
        if batch is not None:
            yield batch['results'][0]


def _simple_download(url, target_file, temp_dir, verbose=3):
    """Wrapper around ``utils._fetch_file``.

    This allows specifying the target file name.

    Parameters
    ----------
    url : str
        URL of the file to download.

    target_file : str
        Location of the downloaded file on filesystem.

    temp_dir : str
        Location of sandbox directory used by ``_fetch_file``.

    verbose : int, optional (default=3)
        an integer in [0, 1, 2, 3] to control the verbosity level.

    Returns
    -------
    target_file : str
        The location in which the file was downloaded.

    Raises
    ------
    URLError, ValueError
        If an error occurred when downloading the file.

    See Also
    --------
    nilearn.datasets._utils._fetch_file


    Notes
    -----
    It can happen that an HTTP error that occurs inside
    ``_fetch_file`` gets transformed into an ``AttributeError`` when
    we try to set the ``reason`` attribute of the exception raised;
    here we replace it with an ``URLError``.

    """
    _print_if('Downloading file: {0}'.format(url), _DEBUG, verbose)
    try:
        downloaded = _fetch_file(url, temp_dir, resume=False,
                                 overwrite=True, verbose=0)
    except Exception as e:
        _print_if('Problem downloading file from {0}'.format(url),
                  _ERROR, verbose)

        # reason is a property of urlib.error.HTTPError objects,
        # but these objects don't have a setter for it, so
        # an HTTPError raised in _fetch_file might be transformed
        # into an AttributeError when we try to set its reason attribute
        if (isinstance(e, AttributeError) and
                e.args[0] == "can't set attribute"):
            raise URLError(
                'HTTPError raised in nilearn.datasets._fetch_file: {0}'.format(
                    traceback.format_exc()))
        raise
    shutil.move(downloaded, target_file)
    _print_if(
        'Download succeeded, downloaded to: {0}'.format(target_file),
        _DEBUG, verbose)
    return target_file


def neurosynth_words_vectorized(word_files, verbose=3, **kwargs):
    """Load Neurosynth data from disk into an (n images, voc size) matrix

    Neurosynth data is saved on disk as ``{word: weight}``
    dictionaries for each image, this function reads it and returns a
    vocabulary list and a term weight matrix.

    Parameters:
    -----------
    word_files : Container
        The paths to the files from which to read word weights (each
        is supposed to contain the Neurosynth response for a
        particular image).

    verbose : int, optional (default=3)
        an integer in [0, 1, 2, 3] to control the verbosity level.

    Keyword arguments are passed on to
    ``sklearn.feature_extraction.DictVectorizer``.

    Returns:
    --------
    frequencies : numpy.ndarray
        An (n images, vocabulary size) array. Each row corresponds to
        an image, and each column corresponds to a word. The words are
        in the same order as in returned value `vocabulary`, so that
        `frequencies[i, j]` corresponds to the weight of
        `vocabulary[j]` for image ``i``.  This matrix is computed by
        an ``sklearn.feature_extraction.DictVectorizer`` instance.

    vocabulary : list of str
        A list of all the words encountered in the word files.

    See Also
    --------
    sklearn.feature_extraction.DictVectorizer

    """
    _print_if('Computing word features.', _INFO, verbose)
    words = []
    voc_empty = True
    for file_name in word_files:
        try:
            with open(file_name, 'rb') as word_file:
                info = json.loads(word_file.read().decode('utf-8'))
                words.append(info['data']['values'])
                if info['data']['values'] != {}:
                    voc_empty = False
        except Exception:
            _print_if(
                'Could not load words from file {0}; error: {1}'.format(
                    file_name, traceback.format_exc()),
                _ERROR, verbose)
            words.append({})
    if voc_empty:
        warnings.warn('No word weight could be loaded, '
                      'vectorizing Neurosynth words failed.')
        return None, None
    vectorizer = DictVectorizer(**kwargs)
    frequencies = vectorizer.fit_transform(words).toarray()
    vocabulary = np.asarray(vectorizer.feature_names_)
    _print_if('Computing word features done; vocabulary size: {0}'.format(
        vocabulary.size), _INFO, verbose)
    return frequencies, vocabulary


def _remove_none_strings(metadata):
    """Replace strings representing a null value with ``None``.

    Some collections and images in Neurovault, for some fields, use the
    string "None", "None / Other", or "null", instead of having ``null``
    in the json file; we replace these strings with ``None`` so that
    they are consistent with the rest and for correct behaviour when we
    want to select or filter out null values.

    Parameters
    ----------
    metadata : dict
        Metadata to transform

    Returns
    -------
    metadata : dict
        Original metadata in which strings representing null values
        have been replaced by ``None``.

    """
    metadata = metadata.copy()
    for key, value in metadata.items():
        if (isinstance(value, _basestring) and
                re.match(r'($|n/?a$|none|null)', value, re.IGNORECASE)):
            metadata[key] = None
    return metadata


def _write_metadata(metadata, file_name):
    """Save metadata to disk.

    Absolute paths are not written; they are recomputed using the
    relative paths when data is loaded again, so that if the
    Neurovault directory has been moved paths are still valid.

    Parameters
    ----------
    metadata : dict
        Dictionary representing metadata for a file or a
        collection. Any key containing 'absolute' is ignored.

    file_name : str
        Path to the file in which to write the data.

    """
    metadata = dict([(k, v) for k, v in metadata.items() if
                     'absolute' not in k])
    with open(file_name, 'wb') as metadata_file:
        metadata_file.write(json.dumps(metadata).encode('utf-8'))


def _add_absolute_paths(root_dir, metadata, force=True):
    """Add absolute paths to a dictionary containing relative paths.

    Parameters
    ----------
    root_dir : str
        The root of the data directory, to prepend to relative paths
        in order to form absolute paths.

    metadata : dict
        Dictionary containing metadata for a file or a collection. Any
        key containing 'relative' is understood to be mapped to a
        relative path and the corresponding absolute path is added to
        the dictionary.

    force : bool, optional (default=True)
        If ``True``, if an absolute path is already present in the
        metadata, it is replaced with the recomputed value. If
        ``False``, already specified absolute paths have priority.

    Returns
    -------
    metadata : dict
        The metadata enriched with absolute paths.

    """
    absolute_paths = {}
    for name, value in metadata.items():
        match = re.match(r'(.*)relative_path(.*)', name)
        if match is not None:
            abs_name = '{0}absolute_path{1}'.format(*match.groups())
            absolute_paths[abs_name] = os.path.join(root_dir, value)
    if not absolute_paths:
        return metadata
    new_metadata = metadata.copy()
    set_func = new_metadata.__setitem__ if force else new_metadata.setdefault
    for name, value in absolute_paths.items():
        set_func(name, value)
    return new_metadata


def _json_from_file(file_name):
    """Load a json file encoded with UTF-8."""
    with open(file_name, 'rb') as dumped:
        loaded = json.loads(dumped.read().decode('utf-8'))
    return loaded


def _json_add_collection_dir(file_name, force=True):
    """Load a json file and add is parent dir to resulting dict."""
    loaded = _json_from_file(file_name)
    set_func = loaded.__setitem__ if force else loaded.setdefault
    dir_path = os.path.dirname(file_name)
    set_func('absolute_path', dir_path)
    set_func('relative_path', os.path.basename(dir_path))
    return loaded


def _json_add_im_files_paths(file_name, force=True):
    """Load a json file and add image and words paths."""
    loaded = _json_from_file(file_name)
    set_func = loaded.__setitem__ if force else loaded.setdefault
    dir_path = os.path.dirname(file_name)
    dir_relative_path = os.path.basename(dir_path)
    image_file_name = 'image_{0}.nii.gz'.format(loaded['id'])
    words_file_name = 'neurosynth_words_for_image_{0}.json'.format(
        loaded['id'])
    set_func('relative_path', os.path.join(dir_relative_path, image_file_name))
    if os.path.isfile(os.path.join(dir_path, words_file_name)):
        set_func('ns_words_relative_path',
                 os.path.join(dir_relative_path, words_file_name))
    loaded = _add_absolute_paths(
        os.path.dirname(dir_path), loaded, force=force)
    return loaded


def _download_collection(collection, download_params):
    """Create directory and download metadata for a collection.

    Parameters
    ----------
    collection : dict
        Collection metadata.

    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Returns
    -------
    collection : dict
        Collection metadata, with local path added to it.

    """
    if collection is None:
        return None
    collection = _remove_none_strings(collection)
    collection_id = collection['id']
    collection_name = 'collection_{0}'.format(collection_id)
    collection_dir = os.path.join(download_params['nv_data_dir'],
                                  collection_name)
    collection['relative_path'] = collection_name
    collection['absolute_path'] = collection_dir
    if not os.path.isdir(collection_dir):
        os.makedirs(collection_dir)
    metadata_file_path = os.path.join(collection_dir,
                                      'collection_metadata.json')
    _write_metadata(collection, metadata_file_path)
    return collection


def _fetch_collection_for_image(image_info, download_params):
    """Find the collection metadata for an image.

    If necessary, the collection metadata is downloaded and its
    directory is created.

    Parameters
    ----------
    image_info : dict
        Image metadata.

    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Returns
    -------
    collection : dict
        The collection metadata.

    """
    collection_id = image_info['collection_id']
    collection_relative_path = 'collection_{0}'.format(collection_id)
    collection_absolute_path = os.path.join(
        download_params['nv_data_dir'], collection_relative_path)
    if not os.path.isdir(collection_absolute_path):
        col_batch = _get_batch(urljoin(
            _NEUROVAULT_COLLECTIONS_URL, str(collection_id)),
            verbose=download_params['verbose'])
        collection = _download_collection(
            col_batch['results'][0], download_params)
    else:
        collection = _json_add_collection_dir(os.path.join(
            collection_absolute_path, 'collection_metadata.json'))

    return collection


def _download_image_nii_file(image_info, collection, download_params):
    """Download an image (.nii.gz) file from Neurovault.

    Parameters
    ----------
    image_info : dict
        Image metadata.

    collection : dict
        Corresponding collection metadata.

    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Returns
    -------
    image_info : dict
        Image metadata with local paths added to it.

    collection : dict
        Corresponding collection metadata with local paths added to it.

    """
    image_info = image_info.copy()
    image_id = image_info['id']
    image_url = image_info['file']
    image_file_name = 'image_{0}.nii.gz'.format(image_id)
    image_relative_path = os.path.join(
        collection['relative_path'], image_file_name)
    image_absolute_path = os.path.join(
        collection['absolute_path'], image_file_name)
    _simple_download(
        image_url, image_absolute_path,
        download_params['temp_dir'], verbose=download_params['verbose'])
    image_info['absolute_path'] = image_absolute_path
    image_info['relative_path'] = image_relative_path
    return image_info, collection


def _check_has_words(file_name):
    if not os.path.isfile(file_name):
        return False
    info = _remove_none_strings(_json_from_file(file_name))
    try:
        assert len(info['data']['values'])
        return True
    except (AttributeError, TypeError, AssertionError):
        pass
    os.remove(file_name)
    return False


def _download_image_terms(image_info, collection, download_params):
    """Download Neurosynth words for an image.

    Parameters
    ----------
    image_info : dict
        Image metadata.

    collection : dict
        Corresponding collection metadata.

    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Returns
    -------
    image_info : dict
        Image metadata with neurosynth words file path added to it.

    collection : dict
        Corresponding collection metadata.

    """

    if not download_params['fetch_neurosynth_words']:
        return image_info, collection

    ns_words_file_name = 'neurosynth_words_for_image_{0}.json'.format(
        image_info['id'])
    image_info = image_info.copy()
    image_info['ns_words_relative_path'] = os.path.join(
        collection['relative_path'], ns_words_file_name)
    image_info['ns_words_absolute_path'] = os.path.join(
        collection['absolute_path'], ns_words_file_name)

    if os.path.isfile(image_info['ns_words_absolute_path']):
        return image_info, collection

    query = urljoin(_NEUROSYNTH_FETCH_WORDS_URL,
                    '?neurovault={0}'.format(image_info['id']))
    try:
        _simple_download(query, image_info['ns_words_absolute_path'],
                         download_params['temp_dir'],
                         verbose=download_params['verbose'])
        assert _check_has_words(image_info['ns_words_absolute_path'])
    except(URLError, ValueError, AssertionError):
        message = 'Could not fetch words for image {0}'.format(
            image_info['id'])
        if not download_params.get('allow_neurosynth_failure', True):
            raise RuntimeError(message)
        _print_if(
            message, _ERROR, download_params['verbose'], with_traceback=True)

    return image_info, collection


def _download_image(image_info, download_params):
    """Download a Neurovault image.

    If necessary, create the corresponding collection's directory and
    download the collection's metadata.

    Parameters
    ----------
    image_info : dict
        Image metadata.

    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Returns
    -------
    image_info : dict
        Image metadata with local paths added to it.

    """
    if image_info is None:
        return None
    image_info = _remove_none_strings(image_info)

    # image_info = self._image_hook(image_info)
    collection = _fetch_collection_for_image(
        image_info, download_params)
    image_info, collection = _download_image_nii_file(
        image_info, collection, download_params)
    image_info, collection = _download_image_terms(
        image_info, collection, download_params)
    metadata_file_path = os.path.join(
            collection['absolute_path'], 'image_{0}_metadata.json'.format(
                image_info['id']))
    _write_metadata(image_info, metadata_file_path)

    return image_info


def _update_image(image_info, download_params):
    """Update local metadata for an image.

    If required and necessary, download the Neurosynth tags.

    Parameters
    ----------
    image_info : dict
        Image metadata.

    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Returns
    -------
    image_info : dict
        Image metadata.

    """
    if not download_params['write_ok']:
        return image_info
    collection = _fetch_collection_for_image(
        image_info, download_params)
    image_info, collection = _download_image_terms(
        image_info, collection, download_params)
    metadata_file_path = os.path.join(
        os.path.dirname(image_info['absolute_path']),
        'image_{0}_metadata.json'.format(image_info['id']))
    _write_metadata(image_info, metadata_file_path)
    return image_info


def _update(image_info, collection, download_params):
    """Update local metadata for an image and its collection."""
    image_info = _update_image(image_info, download_params)
    return image_info, collection


def _scroll_local(download_params):
    """Iterate over local neurovault data.

    Parameters
    ----------
    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the corresponding collection.

    """
    _print_if('Reading local neurovault data.', _DEBUG,
              download_params['verbose'])

    collections = glob(
        os.path.join(
            download_params['nv_data_dir'], '*', 'collection_metadata.json'))

    good_collections = (col for col in
                        (_json_add_collection_dir(col) for col in collections)
                        if download_params['local_collection_filter'](col))
    for collection in good_collections:
        images = glob(os.path.join(
            collection['absolute_path'], 'image_*_metadata.json'))

        good_images = (img for img in
                       (_json_add_im_files_paths(img) for img in images)
                       if download_params['local_image_filter'](img))
        for image in good_images:
            image, collection = _update(image, collection, download_params)
            download_params['visited_images'].add(image['id'])
            download_params['visited_collections'].add(collection['id'])
            yield image, collection


def _scroll_collection(collection, download_params):
    """Iterate over the content of a collection on Neurovault server.

    Images that are found and match filter criteria are downloaded.

    Parameters
    ----------
    collection : dict
        Metadata for the collection

    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Yields
    ------
    image : dict
        Metadata for an image.

    Notes
    -----
    ``image`` can be ``None`` to signify a failed download.

    """
    if collection is None:
        yield None
        return
    n_im_in_collection = 0
    fails_in_collection = 0
    query = urljoin(_NEUROVAULT_COLLECTIONS_URL,
                    '{0}/images/'.format(collection['id']))
    images = _scroll_server_results(
        query, query_terms=download_params['image_terms'],
        local_filter=download_params['image_filter'],
        prefix_msg='Scroll images from collection {0}: '.format(
            collection['id']), batch_size=download_params['batch_size'],
        verbose=download_params['verbose'])

    for image in images:
        if image is None:
            yield None
        try:
            image = _download_image(image, download_params)
            fails_in_collection = 0
            n_im_in_collection += 1
            yield image
        except Exception:
            fails_in_collection += 1
            _print_if(
                '_scroll_collection: bad image: {0}'.format(image),
                _ERROR, download_params['verbose'], with_traceback=True)
            yield None
        if fails_in_collection == download_params['max_fails_in_collection']:
            _print_if('Too many bad images in collection {0}:  '
                      '{1} bad images.'.format(
                          collection['id'], fails_in_collection),
                      _ERROR, download_params['verbose'])
            return
    _print_if(
        'On neurovault.org: '
        '{0} image{1} matched query in collection {2}'.format(
            (n_im_in_collection if n_im_in_collection else 'no'),
            ('s' if n_im_in_collection > 1 else ''), collection['id']),
        _INFO, download_params['verbose'])


def _scroll_filtered(download_params):
    """Iterate over Neurovault data that matches specified filters.

    Images and collections which match the filters provided in the
    download parameters are fetched from the server.

    Parameters
    ----------
    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the corresponding collection.

    Notes
    -----
    ``image``, ``collection`` can be ``None``, ``None`` to signify a
    failed download.

    """
    _print_if('Reading server neurovault data.',
              _DEBUG, download_params['verbose'])

    download_params['collection_filter'] = ResultFilter(
        {'id': NotIn(*download_params['visited_collections'])}).AND(
            download_params['collection_filter'])

    download_params['image_filter'] = ResultFilter(
        {'id': NotIn(*download_params['visited_images'])}).AND(
            download_params['image_filter'])

    collections = _scroll_server_results(
        _NEUROVAULT_COLLECTIONS_URL,
        query_terms=download_params['collection_terms'],
        local_filter=download_params['collection_filter'],
        prefix_msg='Scroll collections: ',
        batch_size=download_params['batch_size'],
        verbose=download_params['verbose'])

    for collection in collections:
        collection = _download_collection(collection, download_params)
        collection_content = _scroll_collection(collection, download_params)
        for image in collection_content:
            yield image, collection


def _scroll_collection_ids(download_params):
    """Download a specific list of collections from Neurovault.

    The collections listed in the download parameters, and all
    the images they contain, are downloaded.

    Parameters
    ----------
    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the corresponding collection.

    Notes
    -----
    ``image``, ``collection`` can be ``None``, ``None`` to signify a
    failed download.

    """
    collection_urls = [
        urljoin(_NEUROVAULT_COLLECTIONS_URL, str(col_id)) for
        col_id in download_params['wanted_collection_ids']]

    if(collection_urls):
        _print_if('Reading server neurovault data.',
                  _DEBUG, download_params['verbose'])

    collections = _yield_from_url_list(
        collection_urls, verbose=download_params['verbose'])
    for collection in collections:
        collection = _download_collection(collection, download_params)
        for image in _scroll_collection(collection, download_params):
            yield image, collection


def _scroll_image_ids(download_params):
    """Download a specific list of images from Neurovault.

    The images listed in the download parameters, and the metadata for
    the collections they belong to, are downloaded.

    Parameters
    ----------
    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the corresponding collection.

    Notes
    -----
    ``image``, ``collection`` can be ``None``, ``None`` to signify a
    failed download.

    """

    image_urls = [urljoin(_NEUROVAULT_IMAGES_URL, str(im_id)) for
                  im_id in download_params['wanted_image_ids']]

    images = _yield_from_url_list(
        image_urls, verbose=download_params['verbose'])
    for image in images:
        try:
            image = _download_image(image, download_params)
            collection = _json_add_collection_dir(os.path.join(
                os.path.dirname(image['absolute_path']),
                'collection_metadata.json'))
        except Exception:
            image, collection = None, None
        yield image, collection


def _scroll_explicit(download_params):
    """Download specific lists of collections and images from Neurovault.

    Parameters
    ----------
    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the corresponding collection.

    Notes
    -----
    ``image``, ``collection`` can be ``None``, ``None`` to signify a
    failed download.

    """

    download_params['wanted_collection_ids'] = set(
        download_params['wanted_collection_ids'] or []).difference(
            download_params['visited_collections'])
    for image, collection in _scroll_collection_ids(download_params):
        if image is not None:
            download_params['visited_images'].add(image['id'])
        yield image, collection

    download_params['wanted_image_ids'] = set(
        download_params['wanted_image_ids'] or []).difference(
            download_params['visited_images'])

    for image, collection in _scroll_image_ids(download_params):
        yield image, collection


def _print_progress(found, download_params, level=_INFO):
    """Print number of images fetched so far."""
    _print_if('Already fetched {0} image{1}'.format(
        found, ('s' if found > 1 else '')),
        level, download_params['verbose'])


def _scroll(download_params):
    """Iterate over Neurovault data.

    Relevant images and collections are loaded from local disk, then
    from neurovault.org

    Parameters
    ----------
    download_params : dict
       General information about download session, containing e.g. the
       data directory (see `_read_download_params` and
       `_prepare_download_params for details`)

    Yields
    ------
    image : dict
        Metadata for an image.

    collection : dict
        Metadata for the corresponding collection.

    Notes
    -----
    Stops if:
        - All available images have been fetched.
        - Or a max number of images has been specified by user and
          reached.
        - Or too many downloads have failed in a row.

    """
    scroll_modes = {'filtered': _scroll_filtered, 'explicit': _scroll_explicit}
    if download_params['max_images'] == 0:
        return
    found = 0

    if download_params['download_mode'] != 'overwrite':
        for image, collection in _scroll_local(download_params):
            found = len(download_params['visited_images'])
            _print_progress(found, download_params, _DEBUG)
            yield image, collection
            if found == download_params['max_images']:
                break
        _print_if('{0} image{1} found on local disk.'.format(
            ('No' if not found else found), ('s' if found > 1 else '')),
            _INFO, download_params['verbose'])

    if download_params['download_mode'] == 'offline':
        return
    if found == download_params['max_images']:
        return
    server_data = scroll_modes[download_params['scroll_mode']](download_params)
    n_consecutive_fails = 0
    for image, collection in server_data:
        if image is None or collection is None:
            n_consecutive_fails += 1
        else:
            n_consecutive_fails = 0
            found += 1
            _print_progress(found, download_params)
            yield image, collection

        if n_consecutive_fails >= download_params['max_consecutive_fails']:
            warnings.warn('Neurovault download stopped early: '
                          'too many downloads failed in a row ({0})'.format(
                              n_consecutive_fails))
            return
        if found == download_params['max_images']:
            return


# Utilities for providing defaults and transforming input and output

def _split_terms(terms, available_on_server):
    """Isolate term filters that can be applied by server."""
    terms_ = dict(terms)
    server_terms = dict([(k, terms_.pop(k)) for k in
                         available_on_server if k in terms_ and
                         (isinstance(terms_[k], _basestring) or
                          isinstance(terms_[k], int))])
    return terms_, server_terms


def _move_unknown_terms_to_local_filter(terms, local_filter,
                                        available_on_server):
    """Move filters handled by the server inside URL.

    Some filters are available on the server and can be inserted into
    the URL query. The rest will have to be applied on metadata
    locally.

    """
    local_terms, server_terms = _split_terms(terms, available_on_server)
    local_filter = ResultFilter(query_terms=local_terms).AND(local_filter)
    return server_terms, local_filter


def basic_collection_terms():
    """Return a term filter that excludes empty collections."""
    return {'number_of_images': NotNull()}


def basic_image_terms():
    """Filter that selects unthresholded F, T and Z maps in mni space

    More precisely, an image is excluded if one of the following is
    true:

        - It is not in MNI space.
        - Its metadata field "is_valid" is cleared.
        - It is thresholded.
        - Its map type is one of "ROI/mask", "anatomical", or "parcellation".
        - Its image type is "atlas"

    """
    return {'not_mni': False, 'is_valid': True, 'is_thresholded': False,
            'map_type': NotIn('ROI/mask', 'anatomical', 'parcellation'),
            'image_type': NotEqual('atlas')}


def _move_col_id(im_terms, col_terms):
    """Reposition 'collection_id' term.

    If the collection id was specified in image filters, move it to
    the collection filters for efficiency.

    This makes specifying the collection id as a keyword argument for
    ``fetch_neurovault`` efficient.

    """
    if 'collection_id' not in im_terms:
        return im_terms, col_terms
    im_terms = copy(im_terms)
    col_terms = copy(col_terms)
    if 'id' not in col_terms:
        col_terms['id'] = im_terms.pop('collection_id')
    elif col_terms['id'] == im_terms['collection_id']:
        col_terms['id'] = im_terms.pop('collection_id')
    else:
        warnings.warn('You specified contradictory collection ids, '
                      'one in the image filters and one in the '
                      'collection filters')
    return im_terms, col_terms


def _read_download_params(
    data_dir, download_mode='download_new', collection_terms=None,
    collection_filter=_empty_filter, image_terms=None,
    image_filter=_empty_filter, wanted_collection_ids=None,
    wanted_image_ids=None, max_images=None,
    max_consecutive_fails=_MAX_CONSECUTIVE_FAILS,
    max_fails_in_collection=_MAX_FAILS_IN_COLLECTION,
    batch_size=None, verbose=3, fetch_neurosynth_words=False,
        vectorize_words=True):

    """Create a dictionary containing download information.

    """
    download_params = {}
    download_params['verbose'] = verbose
    download_mode = download_mode.lower()
    if download_mode not in ['overwrite', 'download_new', 'offline']:
        raise ValueError(
            'supported download modes are overwrite,'
            ' download_new, offline; got {0}'.format(download_mode))
    download_params['download_mode'] = download_mode
    if collection_terms is None:
        collection_terms = {}
    if image_terms is None:
        image_terms = {}
    if max_images is not None and max_images < 0:
        max_images = None
    download_params['nv_data_dir'] = data_dir
    download_params['collection_terms'] = dict(collection_terms)
    download_params['collection_filter'] = collection_filter
    download_params['image_terms'] = dict(image_terms)
    download_params['image_filter'] = image_filter
    download_params['visited_images'] = set()
    download_params['visited_collections'] = set()
    download_params['max_images'] = max_images
    download_params['max_consecutive_fails'] = max_consecutive_fails
    download_params['max_fails_in_collection'] = max_fails_in_collection
    download_params['batch_size'] = batch_size
    download_params['wanted_image_ids'] = wanted_image_ids
    download_params['wanted_collection_ids'] = wanted_collection_ids
    download_params['fetch_neurosynth_words'] = fetch_neurosynth_words
    download_params['write_ok'] = os.access(
        download_params['nv_data_dir'], os.W_OK)
    download_params['vectorize_words'] = vectorize_words
    return download_params


def _prepare_explicit_ids_download_params(download_params):
    """Prepare the download parameters if explicit ids are specified."""
    if download_params.get('wanted_image_ids') is None:
        download_params['wanted_image_ids'] = []
    if download_params.get('wanted_collection_ids') is None:
        download_params['wanted_collection_ids'] = []
    download_params['max_images'] = None
    download_params['scroll_mode'] = 'explicit'
    download_params['image_terms'] = {}
    download_params['image_filter'] = _empty_filter
    download_params['collection_terms'] = {}
    download_params['collection_filter'] = _empty_filter
    download_params['local_collection_filter'] = _empty_filter
    download_params['local_image_filter'] = ResultFilter(
        {'id': IsIn(*download_params['wanted_image_ids'])}).OR(
            ResultFilter(
                collection_id=IsIn(
                    *download_params['wanted_collection_ids'])))
    return download_params


def _prepare_filtered_download_params(download_params):
    """Prepare the download parameters if filters are used."""
    (download_params['image_terms'],
     download_params['collection_terms']) = _move_col_id(
         download_params['image_terms'], download_params['collection_terms'])
    (download_params['collection_terms'],
     download_params['collection_filter']
     ) = _move_unknown_terms_to_local_filter(
         download_params['collection_terms'],
         download_params['collection_filter'],
         _COL_FILTERS_AVAILABLE_ON_SERVER)

    (download_params['image_terms'],
        download_params[
            'image_filter']) = _move_unknown_terms_to_local_filter(
            download_params['image_terms'], download_params['image_filter'],
            _IM_FILTERS_AVAILABLE_ON_SERVER)

    download_params['local_collection_filter'] = ResultFilter(
        **download_params['collection_terms']).AND(
            download_params['collection_filter'])
    download_params['local_image_filter'] = ResultFilter(
        **download_params['image_terms']).AND(
        download_params['image_filter'])

    download_params['scroll_mode'] = 'filtered'
    return download_params


def _prepare_download_params(download_params):
    """Adjust the download parameters.

    Information for the downloaders is added. The result depends on
    whether we are downloading a set of collections and images
    explicitly specified by the user (by id), or we are downloading
    all the collections and images that match certain filters.


    """
    if (download_params['wanted_collection_ids'] is not None or
            download_params['wanted_image_ids'] is not None):
        return _prepare_explicit_ids_download_params(download_params)
    return _prepare_filtered_download_params(download_params)


def _result_list_to_bunch(result_list, download_params):
    """Transform a list of results into a Bunch.

    If necessary, a vocabulary list and a matrix of vectorized tags are
    added.

    """
    if not result_list:
        images_meta, collections_meta = [], []
    else:
        images_meta, collections_meta = zip(*result_list)
        images_meta = list(images_meta)
        collections_meta = list(collections_meta)
    images = [im_meta.get('absolute_path') for im_meta in images_meta]
    result = Bunch(images=images, images_meta=images_meta,
                   collections_meta=collections_meta,
                   description=_get_dataset_descr('neurovault'))
    if download_params[
            'fetch_neurosynth_words'] and download_params['vectorize_words']:
        (result['word_frequencies'],
         result['vocabulary']) = neurosynth_words_vectorized(
             [meta.get('ns_words_absolute_path') for
              meta in images_meta], verbose=download_params['verbose'])
    return result


# High-level functions that provide access to neurovault and neurosynth.
# _fetch_neurovault_implementation does the work, and two interfaces
# are available:
#     fetch_neurovault, to filter results based on metadata
#     fetch_neurovault_ids, to ask for specific images or collections

def _fetch_neurovault_implementation(
    max_images=_DEFAULT_MAX_IMAGES, collection_terms=basic_collection_terms(),
    collection_filter=_empty_filter, image_terms=basic_image_terms(),
    image_filter=_empty_filter, collection_ids=None, image_ids=None,
    mode='download_new', data_dir=None, fetch_neurosynth_words=False,
        vectorize_words=True, verbose=3, **kwarg_image_filters):
    """Download data from neurovault.org and neurosynth.org."""
    image_terms = dict(image_terms, **kwarg_image_filters)
    neurovault_data_dir = _get_dataset_dir('neurovault', data_dir)
    if mode != 'offline' and not os.access(neurovault_data_dir, os.W_OK):
        warnings.warn("You don't have write access to neurovault dir: {0}; "
                      "fetch_neurovault is working offline.".format(
                          neurovault_data_dir))
        mode = 'offline'

    download_params = _read_download_params(
        neurovault_data_dir, download_mode=mode,
        collection_terms=collection_terms,
        collection_filter=collection_filter, image_terms=image_terms,
        image_filter=image_filter, wanted_collection_ids=collection_ids,
        wanted_image_ids=image_ids, max_images=max_images, verbose=verbose,
        fetch_neurosynth_words=fetch_neurosynth_words,
        vectorize_words=vectorize_words)
    download_params = _prepare_download_params(download_params)

    with _TemporaryDirectory() as temp_dir:
        download_params['temp_dir'] = temp_dir
        scroller = list(_scroll(download_params))

    return _result_list_to_bunch(scroller, download_params)


def fetch_neurovault(
    max_images=_DEFAULT_MAX_IMAGES,
    collection_terms=basic_collection_terms(),
    collection_filter=_empty_filter,
    image_terms=basic_image_terms(),
    image_filter=_empty_filter,
    mode='download_new', data_dir=None,
    fetch_neurosynth_words=False, vectorize_words=True,
        verbose=3, **kwarg_image_filters):
    """Download data from neurovault.org that match certain criteria.

    Any downloaded data is saved on the local disk and subsequent
    calls to this function will first look for the data locally before
    querying the server for more if necessary.

    We explore the metadata for Neurovault collections and images,
    keeping those that match a certain set of criteria, until we have
    skimmed through the whole database or until an (optional) maximum
    number of images to fetch has been reached.

    Parameters
    ----------
    max_images : int, optional (default=100)
        Maximum number of images to fetch.

    collection_terms : dict, optional (default=basic_collection_terms())
        Key, value pairs used to filter collection
        metadata. Collections for which
        ``collection_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be discarded.
        See documentation for ``basic_collection_terms`` for a
        description of the default selection criteria.

    collection_filter : Callable, optional (default=_empty_filter)
        Collections for which `collection_filter(collection_metadata)`
        is ``False`` will be discarded.

    image_terms : dict, optional (default=basic_image_terms())
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata['key'] == value`` is not ``True`` for
    if image_filter != _empty_filter and image_terms =
        every key, value pair will be discarded.
        See documentation for ``basic_image_terms`` for a
        description of the default selection criteria.

    image_filter : Callable, optional (default=_empty_filter)
        Images for which `image_filter(image_metadata)` is ``False``
        will be discarded.

    mode : {'download_new', 'overwrite', 'offline'}
        When to fetch an image from the server rather than the local
        disk.

        - 'download_new' (the default) means download only files that
          are not already on disk (regardless of modify date).
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    data_dir : str, optional (default=None)
        The directory we want to use for nilearn data. A subdirectory
        named "neurovault" will contain neurovault data.

    fetch_neurosynth_words : bool, optional (default=False)
        whether to collect words from Neurosynth.

    vectorize_words : bool, optional (default=True)
        If neurosynth words are downloaded, create a matrix of word
        counts and add it to the result. Also add to the result a
        vocabulary list. See ``sklearn.CountVectorizer`` for more info.

    verbose : int, optional (default=3)
        an integer in [0, 1, 2, 3] to control the verbosity level.

    kwarg_image_filters
        Keyword arguments are understood to be filter terms for
        images, so for example ``map_type='Z map'`` means only
        download Z-maps; ``collection_id=35`` means download images
        from collection 35 only.

    Returns
    -------
    Bunch
        A dict-like object which exposes its items as attributes. It contains:

            - 'images', the paths to downloaded files.
            - 'images_meta', the metadata for the images in a list of
              dictionaries.
            - 'collections_meta', the metadata for the
              collections.
            - 'description', a short description of the Neurovault dataset.

        If `fetch_neurosynth_words` and `vectorize_words` were set, it
        also contains:

            - 'vocabulary', a list of words
            - 'word_frequencies', the weight of the words returned by
              neurosynth.org for each image, such that the weight of word
              `vocabulary[j]` for the image found in `images[i]` is
              `word_frequencies[i, j]`

    See Also
    --------
    nilearn.datasets.fetch_neurovault_ids
        Fetch collections and images from Neurovault by explicitly specifying
        their ids.

    Notes
    -----
    Images and collections from disk are fetched before remote data.

    Some helpers are provided in the ``neurovault`` module to express
    filtering criteria more concisely:

        ``ResultFilter``, ``IsNull``, ``NotNull``, ``NotEqual``,
        ``GreaterOrEqual``, ``GreaterThan``, ``LessOrEqual``,
        ``LessThan``, ``IsIn``, ``NotIn``, ``Contains``,
        ``NotContains``, ``Pattern``.

    If you pass a single value to match against the collection id
    (whether as the 'id' field of the collection metadata or as the
    'collection_id' field of the image metadata), the server is
    directly queried for that collection, so
    ``fetch_neurovault(collection_id=40)`` is as efficient as
    ``fetch_neurovault(collection_ids=[40])`` (but in the former
    version the other filters will still be applied). This is not true
    for the image ids. If you pass a single value to match against any
    of the fields listed in ``_COL_FILTERS_AVAILABLE_ON_SERVER``,
    i.e., 'DOI', 'name', and 'owner', these filters can be
    applied by the server, limiting the amount of metadata we have to
    download: filtering on those fields makes the fetching faster
    because the filtering takes place on the server side.

    In `download_new` mode, if a file exists on disk, it is not
    downloaded again, even if the version on the server is newer. Use
    `overwrite` mode to force a new download (you can filter on the
    field ``modify_date`` to re-download the files that are newer on
    the server - see Examples section).

    Tries to yield `max_images` images; stops early if we have fetched
    all the images matching the filters or if too many images fail to
    be downloaded in a row.

    References
    ----------

    .. [1] Gorgolewski KJ, Varoquaux G, Rivera G, Schwartz Y, Ghosh SS,
       Maumet C, Sochat VV, Nichols TE, Poldrack RA, Poline J-B,
       Yarkoni T and Margulies DS (2015) NeuroVault.org: a web-based
       repository for collecting and sharing unthresholded
       statistical maps of the human brain. Front. Neuroinform. 9:8.
       doi: 10.3389/fninf.2015.00008

    .. [2] Yarkoni, Tal, Russell A. Poldrack, Thomas E. Nichols, David
       C. Van Essen, and Tor D. Wager. "Large-scale automated synthesis
       of human functional neuroimaging data." Nature methods 8, no. 8
       (2011): 665-670.

    Examples
    --------
    To download **all** the collections and images from Neurovault::

        fetch_neurovault(max_images=None, collection_terms={}, image_terms={})

    To further limit the default selection to collections which
    specify a DOI (which reference a published paper, as they may be
    more likely to contain good images)::

        fetch_neurovault(
            max_images=None,
            collection_terms=dict(basic_collection_terms(), DOI=NotNull()))

    To update all the images (matching the default filters)::

        fetch_neurovault(
            max_images=None, mode='overwrite',
            modify_date=GreaterThan(newest))

    """
    if max_images == _DEFAULT_MAX_IMAGES:
        _print_if(
            'fetch_neurovault: using default value of {0} for max_images. '
            'Set max_images to another value or None '
            'if you want more images.'.format(_DEFAULT_MAX_IMAGES),
            _INFO, verbose)
    # Users may get confused if they write their image_filter function
    # and the default filters contained in image_terms still apply, so we
    # issue a warning.
    if image_filter != _empty_filter and image_terms == basic_image_terms():
        warnings.warn(
            "You specified a value for `image_filter` but the "
            "default filters in `image_terms` still apply. "
            "If you want to disable them, pass `image_terms={}`")
    if (collection_filter != _empty_filter
            and collection_terms == basic_collection_terms()):
        warnings.warn(
            "You specified a value for `collection_filter` but the "
            "default filters in `collection_terms` still apply. "
            "If you want to disable them, pass `collection_terms={}`")

    return _fetch_neurovault_implementation(
        max_images=max_images, collection_terms=collection_terms,
        collection_filter=collection_filter, image_terms=image_terms,
        image_filter=image_filter, mode=mode,
        data_dir=data_dir,
        fetch_neurosynth_words=fetch_neurosynth_words,
        vectorize_words=vectorize_words, verbose=verbose,
        **kwarg_image_filters)


def fetch_neurovault_ids(
    collection_ids=(), image_ids=(), mode='download_new', data_dir=None,
        fetch_neurosynth_words=False, vectorize_words=True, verbose=3):
    """Download specific images and collections from neurovault.org.

    Any downloaded data is saved on the local disk and subsequent
    calls to this function will first look for the data locally before
    querying the server for more if necessary.

    This is the fast way to get the data from the server if we already
    know which images or collections we want.

    Parameters
    ----------

    collection_ids : Container, optional (default=())
        The ids of whole collections to be downloaded.

    image_ids : Container, optional (default=None)
        The ids of particular images to be downloaded. The metadata for the
        corresponding collections is also downloaded.

    mode : {'download_new', 'overwrite', 'offline'}
        When to fetch an image from the server rather than the local
        disk.

        - 'download_new' (the default) means download only files that
          are not already on disk (regardless of modify date).
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    data_dir : str, optional (default=None)
        The directory we want to use for nilearn data. A subdirectory
        named "neurovault" will contain neurovault data.

    fetch_neurosynth_words : bool, optional (default=False)
        whether to collect words from Neurosynth.

    vectorize_words : bool, optional (default=True)
        If neurosynth words are downloaded, create a matrix of word
        counts and add it to the result. Also add to the result a
        vocabulary list. See ``sklearn.CountVectorizer`` for more info.

    verbose : int, optional (default=3)
        an integer in [0, 1, 2, 3] to control the verbosity level.

    Returns
    -------
    Bunch
        A dict-like object which exposes its items as attributes. It contains:

            - 'images', the paths to downloaded files.
            - 'images_meta', the metadata for the images in a list of
              dictionaries.
            - 'collections_meta', the metadata for the
              collections.
            - 'description', a short description of the Neurovault dataset.

        If `fetch_neurosynth_words` and `vectorize_words` were set, it
        also contains:

            - 'vocabulary', a list of words
            - 'word_frequencies', the weight of the words returned by
              neurosynth.org for each image, such that the weight of word
              `vocabulary[j]` for the image found in `images[i]` is
              `word_frequencies[i, j]`

    See Also
    --------
    nilearn.datasets.fetch_neurovault
        Fetch data from Neurovault, but use filters on metadata to select
        images and collections rather than giving explicit lists of ids.

    Notes
    -----
    Images and collections from disk are fetched before remote data.

    In `download_new` mode, if a file exists on disk, it is not
    downloaded again, even if the version on the server is newer. Use
    `overwrite` mode to force a new download.

    Stops early if too many images fail to be downloaded in a row.

    References
    ----------

    .. [1] Gorgolewski KJ, Varoquaux G, Rivera G, Schwartz Y, Ghosh SS,
       Maumet C, Sochat VV, Nichols TE, Poldrack RA, Poline J-B,
       Yarkoni T and Margulies DS (2015) NeuroVault.org: a web-based
       repository for collecting and sharing unthresholded
       statistical maps of the human brain. Front. Neuroinform. 9:8.
       doi: 10.3389/fninf.2015.00008

    .. [2] Yarkoni, Tal, Russell A. Poldrack, Thomas E. Nichols, David
       C. Van Essen, and Tor D. Wager. "Large-scale automated synthesis
       of human functional neuroimaging data." Nature methods 8, no. 8
       (2011): 665-670.

    """
    return _fetch_neurovault_implementation(
        mode=mode,
        collection_ids=collection_ids, image_ids=image_ids,
        data_dir=data_dir,
        fetch_neurosynth_words=fetch_neurosynth_words,
        vectorize_words=vectorize_words, verbose=verbose)
