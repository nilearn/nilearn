"""
Download statistical maps available on Neurovault (http://neurovault.org).
"""
# Author: Jerome Dockes
# License: simplified BSD

import os
import logging
import warnings
from copy import copy, deepcopy
import shutil
import re
import json
from glob import glob
from tempfile import mkdtemp
from collections import Container
try:
    from collections import OrderedDict
except ImportError:
    # using python2.6
    OrderedDict = dict
import traceback
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


class MaxImagesReached(StopIteration):
    """Exception class to signify enough images have been fetched."""
    pass


def prepare_logging(level=logging.DEBUG):
    """Get the root logger.

    Parameters
    ----------
    level : int, optional (default=logging.DEBUG)
        Level of the handler that is added if none exist.
        this handler streams output to the console.

    Returns
    -------
    logging.RootLogger
        The root logger.

    """
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


_logger = prepare_logging(level=logging.INFO)


def add_logging_to_console(level=logging.DEBUG):
    console_logger = logging.StreamHandler()
    console_logger.setLevel(level)
    formatter = logging.Formatter('%(message)s')
    console_logger.setFormatter(formatter)
    _logger.addHandler(console_logger)


def add_log_file(file_name, level=logging.DEBUG):
    file_logger = logging.FileHandler(file_name)
    file_logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    file_logger.setFormatter(formatter)
    _logger.addHandler(file_logger)


def set_logging_level(level=logging.INFO):
    for handler in _logger.handlers:
        handler.setLevel(level)



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


def _empty_filter(arg):
    """Place holder for a filter which always returns True."""
    return True


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


def _get_batch(query, prefix_msg='', timeout=10.):
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
    _logger.debug('{0}getting new batch: {1}'.format(
        prefix_msg, query))
    try:
        resp = opener.open(request, timeout=timeout)

    except Exception:
        _logger.exception(
            'Could not download batch from {0}'.format(query))
        raise
    try:
        encoding = _get_encoding(resp)
        content = resp.read()
        batch = json.loads(content.decode(encoding))
    except(URLError, ValueError):
        _logger.exception('Could not decypher batch from {0}'.format(query))
        raise
    finally:
        resp.close()
    if 'id' in batch:
        batch = {'count': 1, 'results': [batch]}
    for key in ['results', 'count']:
        if batch.get(key) is None:
            msg = ('Could not find required key "{0}" '
                   'in batch retrieved from {1}'.format(key, query))
            _logger.error(msg)
            raise ValueError(msg)

    return batch


def _scroll_server_results(url, local_filter=_empty_filter,
                           query_terms=None, max_results=None,
                           batch_size=None, prefix_msg=''):
    """Download list of metadata from Neurovault.

    Parameters
    ----------
    url : str
        The base url (without the filters) from which to get data.

    local_filter : callable, optional (default=_empty_filter)
        Used to filter the results based on their metadata:
        must return True is the result is to be kept and False otherwise.
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

    Yields
    ------
    result : dict
        A result in the retrieved batch.

    Raises
    ------
    URLError, ValueError
        If a batch failed to be retrieved.

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
        batch = _get_batch(new_query, prefix_msg)
        batch_size = len(batch['results'])
        downloaded += batch_size
        _logger.debug('{0}batch size: {1}'.format(prefix_msg, batch_size))
        if n_available is None:
            n_available = batch['count']
            max_results = (n_available if max_results is None
                           else min(max_results, n_available))
        for result in batch['results']:
            if local_filter(result):
                yield result


def _yield_from_url_list(url_list):
    """Get metadata coming from an explicit list of URLs.

    This is different from ``_scroll_server_results``, which is used
    to get all the metadata that matches certain filters.

    Parameters
    ----------
    url_list : Container of str
        URLs from which to get data

    Yields
    ------
    content : dict
        The metadata from one URL.

    Raises
    ------
    urllib.error.URLError
        If there was a problem opening an URL.

    ValueError
        If a response could not be decoded, or did not contain either
        'id' (single result), or 'results' and 'count' (batch).

    """
    for url in url_list:
        content = _get_batch(url)['results'][0]
        yield content


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

    """
    def __eq__(self, other):
        return bool(other)


class NotEqual(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with `NotEqual(obj)`. It
    will allways be equal to, and only to, any value for which
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
    will allways be equal to, and only to, any value for which
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

    """
    def _eq_impl(self, other):
        return self.bound_ <= other


class GreaterThan(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `GreaterThan(obj)`. It
    will allways be equal to, and only to, any value for which
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

    """
    def _eq_impl(self, other):
        return self.bound_ < other


class LessOrEqual(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `LessOrEqual(obj)`. It
    will allways be equal to, and only to, any value for which
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

    """
    def _eq_impl(self, other):
        return other <= self.bound_


class LessThan(_OrderComp):
    """Special value used to filter terms.

    An instance of this class is constructed with `LessThan(obj)`. It
    will allways be equal to, and only to, any value for which
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

    """
    def _eq_impl(self, other):
        return other < self.bound_


class IsIn(_SpecialValue):
    """Special value used to filter terms.

    An instance of this class is constructed with
    `IsIn(*accepted)`. It will allways be equal to, and only to, any
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
    >>> countable = IsIn(*range(11))
    >>> 7 == countable
    True
    >>> countable == 12
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
    `NotIn(*rejected)`. It will allways be equal to, and only to, any
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
    `Contains(*must_be_contained)`. It will allways be equal to, and
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
    `NotContains(*must_not_be_contained)`. It will allways be equal
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

    `Pattern(pattern[, flags])`. It will allways be equal to, and only
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
    >>> pattern = Pattern(r'[0-9akqj]{5}$')
    >>> 'ak05q' == pattern
    True
    >>> 'ak05e' == pattern
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
        """Remove item from query_terms"""
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


def _simple_download(url, target_file, temp_dir):
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
    print('Downloading file {} to {}'.format(url, target_file))
    _logger.debug('Downloading file: {0}'.format(url))
    try:
        downloaded = _fetch_file(url, temp_dir, resume=False,
                                 overwrite=True, verbose=0)
    except Exception as e:
        _logger.error('Problem downloading file from {0}'.format(url))

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
    _logger.debug(
        'Download succeeded, downloaded to: {0}'.format(target_file))
    return target_file


# def _checked_get_dataset_dir(dataset_name, suggested_dir=None,
#                              write_required=False):
#     """Wrapper for ``_get_dataset_dir``.
#
#     Expands . and ~ and checks write access.
#
#     Parameters
#     ----------
#     dataset_name : str
#         Passed to ``_get_dataset_dir``. Example: ``neurovault``.
#
#     suggested_dir : str
#         Desired location of root data directory for all datasets,
#         e.g. ``~/home/nilearn_data``.
#
#     write_required : bool, optional (default=False)
#         If ``True``, check that the user has write access to the
#         chosen data directory and raise ``IOError`` if not.  If
#         ``False``, don't check for write permission.
#
#     Returns
#     -------
#     dataset_dir : str
#         The location of the dataset directory in the filesystem.
#
#     Raises
#     ------
#     IOError
#         If `write_required` is set and the user doesn't have write
#         access to `dataset_dir`.
#
#     See Also
#     --------
#     nilearns.datasets._utils._get_dataset_dir
#
#     """
#     if suggested_dir is not None:
#         suggested_dir = os.path.abspath(os.path.expanduser(suggested_dir))
#     dataset_dir = _get_dataset_dir(dataset_name, data_dir=suggested_dir)
#     if not write_required:
#         return dataset_dir
#     if not os.access(dataset_dir, os.W_OK):
#         raise IOError('Permission denied: {0}'.format(dataset_dir))
#     return dataset_dir
#
#
# def neurovault_directory(suggested_dir=None):
#     """Return path to neurovault directory on filesystem."""
#     if getattr(neurovault_directory, 'directory_path_', None) is not None:
#         return neurovault_directory.directory_path_
#
#     _logger.debug('Looking for Neurovault directory.')
#     if suggested_dir is None:
#         root_data_dir, dataset_name = None, 'neurovault'
#     else:
#         suggested_path = suggested_dir.split(os.path.sep)
#         dataset_name = suggested_path[-1]
#         root_data_dir = os.path.sep.join(suggested_path[:-1])
#     neurovault_directory.directory_path_ = _checked_get_dataset_dir(
#         dataset_name, root_data_dir)
#     assert(neurovault_directory.directory_path_ is not None)
#     _logger.debug('Found Neurovault directory in {0}'.format(
#         neurovault_directory.directory_path_))
#     return neurovault_directory.directory_path_
#
#
# def set_neurovault_directory(new_neurovault_dir=None):
#     """Set the default neurovault directory to a new location.
#
#     Parameters
#     ----------
#     new_neurovault_dir : str, optional (default=None)
#         Suggested path for neurovault directory.
#         The default value ``None`` means reset neurovault directory
#         path to its default value.
#
#     Returns
#     -------
#
#     neurovault_directory.directory_path_ : str
#         The new neurovault directory used by default by all functions.
#
#     See Also
#     --------
#     nilearn.datasets.neurovault.neurovault_directory
#
#     """
#     _logger.debug('Set neurovault directory: {0}...'.format(
#         new_neurovault_dir))
#     neurovault_directory.directory_path_ = None
#     return neurovault_directory(new_neurovault_dir)


def _get_temp_dir(suggested_dir=None):
    """Get a sandbox dir in which to download files."""
    if suggested_dir is not None:
        suggested_dir = os.path.abspath(os.path.expanduser(suggested_dir))
    if (suggested_dir is None or
        not os.path.isdir(suggested_dir) or
        not os.access(suggested_dir, os.W_OK)):
        suggested_dir = mkdtemp()
    return suggested_dir


def _fetch_neurosynth_words(image_id, target_file, temp_dir):
    """Query Neurosynth for words associated with a map.

    Parameters
    ----------
    image_id : int
        The Neurovault id of the statistical map.

    target_file : str
        Path to the file in which the terms will be stored on disk
        (a json file).

    temp_dir : str
        Path to directory used by ``_simple_download``.

    Returns
    -------
    None

    """
    query = urljoin(_NEUROSYNTH_FETCH_WORDS_URL,
                    '?neurovault={0}'.format(image_id))
    _simple_download(query, target_file, temp_dir)


def neurosynth_words_vectorized(word_files, **kwargs):
    """Load Neurosynth data from disk into an (n files, voc size) matrix

    Neurosynth data is saved on disk as ``{word: weight}``
    dictionaries for each image, this function reads it and returs a
    vocabulary list and a term weight matrix.

    Parameters:
    -----------
    word_files : container
        The paths to the files from which to read word weights (each
        is supposed to contain the Neurosynth response for a
        particular image).

    Keyword arguments are passed on to
    ``sklearn.feature_extraction.DictVectorizer``.

    Returns:
    --------
    vocabulary : list of str
        A list of all the words encountered in the word files.

    frequencies : numpy.ndarray
        An (n images, vocabulary size) array. Each row corresponds to
        an image, and each column corresponds to a word. The words are
        in the same order as in returned vaule `vocabulary`, so that
        `frequencies[i, j]` corresponds to the weight of
        `vocabulary[j]` for image ``i``.  This matrix is computed by
        an ``sklearn.feature_extraction.DictVectorizer`` instance.

    See Also
    --------
    sklearn.feature_extraction.DictVectorizer

    """
    _logger.info('Computing word features.')
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
            _logger.warning(
                'Could not load words from file {0}; error: {1}'.format(
                    file_name, traceback.format_exc()))
            words.append({})
    if voc_empty:
        _logger.warning('No word weight could be loaded, '
                        'vectorizing Neurosynth words failed.')
        return None, None
    vectorizer = DictVectorizer(**kwargs)
    frequencies = vectorizer.fit_transform(words).toarray()
    vocabulary = np.asarray(vectorizer.feature_names_)
    _logger.info('Computing word features done; vocabulary size: {0}'.format(
        vocabulary.size))
    return frequencies, vocabulary


def _remove_none_strings(metadata):
    """Replace strings representing a null value with ``None``.

    Some collections and images in Neurovault, for some fields, use
    the string "None", "None / Other", or "null", instead of having
    ``null`` in the json file; we replace these strings with None so
    that they are consistent with the rest and for correct behaviour
    when we want to select or filter out null values.

    """
    for key, value in metadata.items():
        if (isinstance(value, _basestring) and
            re.match(r'($|n/?a$|none|null)', value, re.IGNORECASE)):
            metadata[key] = None
    return metadata


class BaseDownloadManager(object):
    """Base class for all Neurovault download managers.

    download managers are used as parameters for fetch_neurovault;
    they download the files and store them on disk.

    A ``BaseDownloadManager`` does not download anything, but
    increments a counter and raises a ``MaxImagesReached`` exception
    when the specified max number of images has been reached.

    Subclasses should override ``_collection_hook`` and
    ``_image_hook`` in order to perform the actual work.  They should
    not override ``image`` as it is responsible for stopping the
    stream of metadata when the max numbers of images has been
    reached.

    Parameters
    ----------
    neurovault_data_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. This is
        passed on to _get_dataset_dir, which may result in another
        directory being used if the one that was specified is not
        valid.

    max_images : int, optional(default=100)
        Maximum number of images to fetch. ``None`` or a negative
        value means download as many as you can.

    """
    def __init__(self, neurovault_data_dir, max_images=100):
        self.nv_data_dir_ = neurovault_data_dir
        if max_images is not None and max_images < 0:
            max_images = None
        self.max_images_ = max_images
        self.already_downloaded_ = 0
        self.write_ok_ = os.access(self.nv_data_dir_, os.W_OK)

    def collection(self, collection_info):
        """Receive metadata for a collection and take necessary actions.

        The actual work is delegated to ``self._collection_hook``,
        which subclasses should override.

        """
        collection_info = _remove_none_strings(collection_info)
        return self._collection_hook(collection_info)

    def image(self, image_info):
        """Receive metadata for an image and take necessary actions.

        Stop metadata stream if maximum number of images has been
        reached.

        The actual work is delegated to ``self._image_hook``,
        which subclasses should override.

        """
        if self.already_downloaded_ == self.max_images_:
            raise MaxImagesReached()
        if image_info is None:
            return None
        image_info = _remove_none_strings(image_info)
        image_info = self._image_hook(image_info)
        if image_info is not None:
            self.already_downloaded_ += 1
        return image_info

    def update_image(self, image_info):
        return image_info

    def update_collection(self, collection_info):
        return collection_info

    def update(self, image_info, collection_info):
        """Act when metadata stored on disk is seen again."""
        image_info = self.update_image(image_info)
        collection_info = self.update_collection(collection_info)
        return image_info, collection_info

    def _collection_hook(self, collection_info):
        """Hook for subclasses."""
        return collection_info

    def _image_hook(self, image_info):
        """Hook for subclasses."""
        return image_info

    def start(self):
        """Prepare for download session."""
        pass

    def finish(self):
        """Cleanup after download session."""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.finish()


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
    set_func = metadata.__setitem__ if force else metadata.setdefault
    absolute_paths = {}
    for name, value in metadata.items():
        match = re.match(r'(.*)relative_path(.*)', name)
        if match is not None:
            abs_name = '{0}absolute_path{1}'.format(*match.groups())
            absolute_paths[abs_name] = os.path.join(root_dir, value)
    for name, value in absolute_paths.items():
        set_func(name, value)
    return metadata


def _re_raise(error):
    raise(error)


def _tolerate_failure(error):
    pass


class DownloadManager(BaseDownloadManager):
    """Store maps, metadata, reduced representations and associated words.

    For each collection, this download manager creates a subdirectory
    in the Neurovault directory and stores in it:

        - Metadata for the collection (in .json files).
        - Metadata for the brain maps (in .json files), the brain maps
          (in .nii.gz files).
        - Optionally, the reduced representations of the brain maps
          (in .npy files).
        - Optionally, for each image, the words weights associated to
          it by Neurosynth.

    Parameters
    ----------
    neurovault_data_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. This is
        passed on to _get_dataset_dir, which may result in another
        directory being used if the one that was specified is not
        valid.

    max_images : int, optional(default=100)
        Maximum number of images to fetch. ``None`` or a negative
        value means download as many as you can.

    temp_dir : str or None, optional (default=None)
        Sandbox directory for downloads.  if None, a temporary
        directory is created by ``tempfile.mkdtemp``.

    fetch_neurosynth_words : bool, optional (default=False)
        Wether to collect words from Neurosynth.

    fetch_reduced_rep : bool, optional (default=True)
        Wether to download the reduced representations from
        Neurovault.

    neurosynth_error_handler :
        Callable, optional (default=_tolerate_failure)
        What to do when words for an image could not be
        retrieved. The default value keeps the image anyway and
        does not raise an error.

    """
    def __init__(self, neurovault_data_dir=None, temp_dir=None,
                 fetch_neurosynth_words=False, fetch_reduced_rep=True,
                 max_images=100, neurosynth_error_handler=_tolerate_failure):

        super(DownloadManager, self).__init__(
            neurovault_data_dir=neurovault_data_dir, max_images=max_images)
        self.suggested_temp_dir_ = temp_dir
        self.temp_dir_ = None
        self.fetch_ns_ = fetch_neurosynth_words
        self.fetch_reduced_rep_ = fetch_reduced_rep
        self.neurosynth_error_handler_ = neurosynth_error_handler

    def _collection_hook(self, collection_info):
        """Create collection subdir and store metadata.

        Parameters
        ----------
        collection_info : dict
            Collection metadata

        Returns
        -------
        collection_info : dict
            Collection metadata, with local path to collection
            subdirectory added to it.

        """
        collection_id = collection_info['id']
        collection_name = 'collection_{0}'.format(collection_id)
        collection_dir = os.path.join(self.nv_data_dir_, collection_name)
        collection_info['relative_path'] = collection_name
        collection_info['absolute_path'] = collection_dir
        if not os.path.isdir(collection_dir):
            os.makedirs(collection_dir)
        metadata_file_path = os.path.join(collection_dir,
                                          'collection_metadata.json')
        _write_metadata(collection_info, metadata_file_path)
        return collection_info

    def _add_words(self, image_info):
        """Get the Neurosynth words for an image and write them to disk.

        If ``self.fetch_ns_ is ``False``, nothing is done.
        Errors that occur when fetching words from Neurosynth are
        handled by ``self.neurosynth_error_handler_``.
        If the corresponding file already exists on disk, the server
        is not queryied again.

        Parameters
        ----------
        image_info : dict
            Image metadata.

        Returns
        -------
        image_info : dict
            Image metadata, with local paths to image, reduced
            representation (if fetched), and Neurosynth words (if
            fetched) added to it.

        """
        if not self.fetch_ns_:
            return image_info

        collection_absolute_path = os.path.dirname(
            image_info['absolute_path'])
        collection_relative_path = os.path.basename(
            collection_absolute_path)
        ns_words_file_name = 'neurosynth_words_for_image_{0}.json'.format(
            image_info['id'])
        ns_words_relative_path = os.path.join(collection_relative_path,
                                              ns_words_file_name)
        ns_words_absolute_path = os.path.join(collection_absolute_path,
                                              ns_words_file_name)
        if not os.path.isfile(ns_words_absolute_path):
            try:
                _fetch_neurosynth_words(
                    image_info['id'],
                    ns_words_absolute_path, self.temp_dir_)
            except(URLError, ValueError) as e:
                _logger.exception(
                    'Could not fetch words for image {0}'.format(
                        image_info['id']))
                self.neurosynth_error_handler_(e)
                return
        image_info[
            'neurosynth_words_relative_path'] = ns_words_relative_path
        image_info[
            'neurosynth_words_absolute_path'] = ns_words_absolute_path
        return image_info

    def _image_hook(self, image_info):
        """Download image, reduced representation, Neurosynth words.

        Wether reduced representation and Neurosynth words are
        downloaded depends on ``self.fetch_reduced_rep_`` and
        ``self.fetch_ns_``. If there is no matching collection
        directory and metadata on disk, the collection directory is
        created and the metadata is downloaded.

        Parameters
        ----------
        image_info: dict
            Image metadata.

        Returns
        -------
        image_info: dict
            Image metadata, with local path to image, local path to
            reduced representation (if reduced representation
            available and ``self.fetch_reduced_rep_``), and local path
            to Neurosynth words (if ``self.fetch_ns_``) added to it.

        """
        collection_id = image_info['collection_id']
        collection_relative_path = 'collection_{0}'.format(collection_id)
        collection_absolute_path = os.path.join(
            self.nv_data_dir_, collection_relative_path)
        if not os.path.isdir(collection_absolute_path):
            col_batch = _get_batch(urljoin(
                _NEUROVAULT_COLLECTIONS_URL, str(collection_id)))
            self.collection(col_batch['results'][0])
        image_id = image_info['id']
        image_url = image_info['file']
        image_file_name = 'image_{0}.nii.gz'.format(image_id)
        image_relative_path = os.path.join(
            collection_relative_path, image_file_name)
        image_absolute_path = os.path.join(
            collection_absolute_path, image_file_name)
        _simple_download(image_url, image_absolute_path, self.temp_dir_)
        image_info['absolute_path'] = image_absolute_path
        image_info['relative_path'] = image_relative_path
        reduced_image_url = image_info.get('reduced_representation')
        if self.fetch_reduced_rep_ and reduced_image_url is not None:
            reduced_image_name = 'image_{0}_reduced_rep.npy'.format(image_id)
            reduced_image_relative_path = os.path.join(
                collection_relative_path, reduced_image_name)
            reduced_image_absolute_path = os.path.join(
                collection_absolute_path, reduced_image_name)
            _simple_download(
                reduced_image_url, reduced_image_absolute_path, self.temp_dir_)
            image_info['reduced_representation'
                       '_relative_path'] = reduced_image_relative_path
            image_info['reduced_representation'
                       '_absolute_path'] = reduced_image_absolute_path
        image_info = self._add_words(image_info)
        metadata_file_path = os.path.join(
            collection_absolute_path, 'image_{0}_metadata.json'.format(
                image_id))
        _write_metadata(image_info, metadata_file_path)
        # self.already_downloaded_ is incremented only after
        # this routine returns successfully.
        print('Already fetched {0:6<} image{1}'.format(
            self.already_downloaded_ + 1,
            ('s' if self.already_downloaded_ + 1 > 1 else '')))
        _logger.info('Already fetched {0} image{1}.'.format(
            self.already_downloaded_ + 1,
            ('s' if self.already_downloaded_ + 1 > 1 else '')))
        return image_info

    def update_image(self, image_info):
        """Download Neurosynth words if necessary.

        If ``self.fetch_ns_`` is set and Neurosynth words are not on
        disk, fetch them and add their location to image metadata.

        """
        if not self.write_ok_:
            return image_info
        image_info = self._add_words(image_info)
        metadata_file_path = os.path.join(
            os.path.dirname(image_info['absolute_path']),
            'image_{0}_metadata.json'.format(image_info['id']))
        _write_metadata(image_info, metadata_file_path)
        return image_info

    def start(self):
        """Prepare for a download session.

        If we don't have a sandbox directory for downloads, create
        one.

        """
        if self.temp_dir_ is None:
            self.temp_dir_ = _get_temp_dir(self.suggested_temp_dir_)

    def finish(self):
        """Cleanup after download session.

        If ``self.start`` created a temporary directory for the
        download session, remove it.

        """
        if self.temp_dir_ is None:
            return
        if self.temp_dir_ != self.suggested_temp_dir_:
            shutil.rmtree(self.temp_dir_)
            self.temp_dir_ = None


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
    """Load a json file and add image, reduced rep and words paths."""
    loaded = _json_from_file(file_name)
    set_func = loaded.__setitem__ if force else loaded.setdefault
    dir_path = os.path.dirname(file_name)
    dir_relative_path = os.path.basename(dir_path)
    image_file_name = 'image_{0}.nii.gz'.format(loaded['id'])
    reduced_file_name = 'image_{0}_reduced_rep.npy'.format(loaded['id'])
    words_file_name = 'neurosynth_words_for_image_{0}.json'.format(
        loaded['id'])
    set_func('relative_path', os.path.join(dir_relative_path, image_file_name))
    if os.path.isfile(os.path.join(dir_path, reduced_file_name)):
        set_func('reduced_representation_relative_path',
                 os.path.join(dir_relative_path, reduced_file_name))
    if os.path.isfile(os.path.join(dir_path, words_file_name)):
        set_func('neurosynth_words_relative_path',
                 os.path.join(dir_relative_path, words_file_name))
    loaded = _add_absolute_paths(
        os.path.dirname(dir_path), loaded, force=force)
    return loaded


def _managed_method(attr='download_manager_'):
    def wrapper(fun):
        def wrapped(*args, **kwargs):
            with getattr(args[0], attr):
                for item in fun(*args, **kwargs):
                    yield item
        return wrapped
    return wrapper


class _DataScroller(object):
    """Iterate over neurovault.org results for a query.

    Parameters
    ----------
    collection_terms : dict, optional (default=None)
        Key, value pairs used to filter collection
        metadata. Collections for which
        ``collection_metadata[key] == value`` is not ``True``
        for every key, value pair will be ignored.
        If ``None``, the empty dictionary is used.

    collection_filter : Callable, optional (default=_empty_filter)
        Collections for which
        `collection_filter(collection_metadata)` is ``False``
        will be ignored.

    image_terms : dict, optional (default=None)
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata[key] == value`` is not ``True`` for
        every key, value pair will be ignored.
        If ``None``, the empty dictionary is used.

    image_filter : Callable, optional (default=_empty_filter)
        Images for which `image_filter(image_metadata)` is
        ``False`` will be ignored.

    ignored_collection_ids : Container, optional (default=None)
        Collections we already have and should not download.
        If ``None``, the empty set is used.

    ignored_image_ids : Container, optional (default=None)
        Images we already have and should not download.
        If ``None``, the empty set is used.

    wanted_collection_ids : Container, optional (default=None)
        If not ``None``, ignore filters and terms and just download
        all collections from this list (and if specified all the
        images from `wanted_image_ids`)

    wanted_image_ids : Container, optional (default=None)
        If not ``None``, ignore filters and terms and just download
        all images from this list (and if specified all the
        collections from `wanted_collection_ids`)

    download_manager : BaseDownloadManager, optional (default=None)
        The download manager used to handle data from neurovault.org.
        If ``None``, one is constructed.

    max_images : int, optional (default=None)
        Maximum number of images to download; only used if
        `download_manager` is None. Ignored if an explicit list of
        collections (resp. images) has been specified with
        `wanted_collection_ids` (resp. `wanted_image_ids`).

    max_consecutive_fails : int, optional (default=10)
        If more than `max_consecutive_fails` images or collections in
        a row fail to be downloaded, we consider there is a problem
        and stop the download session. Does not aplly to explicitely
        specified images: the download will stop with an error message
        if an image explicitly required by the caller fails to be
        downloaded.

    max_fails_in_collection : int, optional (default=5)
        If more than `max_fails_in_collection` images fail to be
        downloaded from a collection, we consider this collection is
        bad and move on to the next collection.

    batch_size : int, optional (default=None)
        Neurovault sends metadata in batches. `batch_size` is the size
        of the batches to ask for. If ``None``, the default
        ``_DEFAULT_BATCH_SIZE`` will be used.

    """
    def __init__(self, data_dir, download_mode='download_new',
                 collection_terms=None, collection_filter=_empty_filter,
                 image_terms=None, image_filter=_empty_filter,
                 wanted_collection_ids=None, wanted_image_ids=None,
                 download_manager=None, max_images=None,
                 max_consecutive_fails=10, max_fails_in_collection=5,
                 batch_size=None):
        download_mode = download_mode.lower()
        if download_mode not in ['overwrite', 'download_new', 'offline']:
            raise ValueError(
                'supported download modes are overwrite,'
                ' download_new, offline; got {0}'.format(download_mode))
        self.download_mode_ = download_mode
        self.neurovault_dir_ = data_dir
        if collection_terms is None:
            collection_terms = {}
        if image_terms is None:
            image_terms = {}
        if max_images is not None and max_images < 0:
            max_images = None
        self.collection_terms_ = collection_terms
        self.collection_filter_ = collection_filter
        self.image_terms_ = image_terms
        self.image_filter_ = image_filter
        self.visited_images_, self.visited_collections_ = set(), set()
        self.max_images_ = max_images
        self.max_consecutive_fails_ = max_consecutive_fails
        self.max_fails_in_collection_ = max_fails_in_collection
        self.consecutive_fails_ = 0
        self.fails_in_collection_ = 0
        self.batch_size_ = batch_size

        if download_manager is None:
            download_manager = BaseDownloadManager(
                neurovault_data_dir=self.neurovault_dir_,
                max_images=self.max_images_)
        self.download_manager_ = download_manager
        self.download_manager_.neurovault_data_dir_ = self.neurovault_dir_
        self.wanted_image_ids_ = wanted_image_ids
        self.wanted_collection_ids_ = wanted_collection_ids
        self._prepare()

    def _prepare(self):
        if (self.wanted_collection_ids_ is not None or
            self.wanted_image_ids_ is not None):

            self.max_images_ = None
            self.download_manager_.max_images_ = None
            self.scroll_mode_ = 'explicit'
            self.image_terms_ = {}
            self.image_filter_ = _empty_filter
            self.collection_terms_ = {}
            self.collection_filter_ = _empty_filter
            self.local_collection_filter_ = _empty_filter
            self.local_image_filter_ = ResultFilter(
                {'id': IsIn(*self.wanted_image_ids_)}).OR(
                    ResultFilter(
                        collection_id=IsIn(*self.wanted_collection_ids_)))

            return

        (self.collection_terms_,
         self.collection_filter_) = _move_unknown_terms_to_local_filter(
             self.collection_terms_, self.collection_filter_,
             _COL_FILTERS_AVAILABLE_ON_SERVER)

        (self.image_terms_,
         self.image_filter_) = _move_unknown_terms_to_local_filter(
             self.image_terms_, self.image_filter_,
             _IM_FILTERS_AVAILABLE_ON_SERVER)

        self.local_collection_filter_ = ResultFilter(
            **self.collection_terms_).AND(self.collection_filter_)
        self.local_image_filter_ = ResultFilter(**self.image_terms_).AND(
            self.image_filter_)

        self.scroll_mode_ = 'filtered'

    def _failed_download(self):
        self.consecutive_fails_ += 1
        if self.consecutive_fails_ == self.max_consecutive_fails_:
            _logger.error('Too many failed downloads: {}; '
                          'stop scrolling remote data.'.format(
                              self.consecutive_fails_))
            raise RuntimeError(
                '{0} consecutive bad downloads'.format(
                    self.consecutive_fails_))

    def _scroll_collection(self, collection):
        """Iterate over the content of a collection on Neurovault server.

        Parameters
        ----------
        collection : dict
            The collection metadata.

        Yields
        ------
        image : dict
            Metadata for an image.

        Raises
        ------
        MaxImagesReached
            If enough images have been downloaded.

        RuntimeError
            If more than ``self.max_consecutive_fails`` images have
            failed in a row.

        """
        n_im_in_collection = 0
        fails_in_collection = 0
        query = urljoin(_NEUROVAULT_COLLECTIONS_URL,
                        '{0}/images/'.format(collection['id']))
        images = _scroll_server_results(
            query, query_terms=self.image_terms_,
            local_filter=self.image_filter_,
            prefix_msg='Scroll images from collection {0}: '.format(
                collection['id']),
            batch_size=self.batch_size_)
        while True:
            image = None
            try:
                image = next(images)
                image = self.download_manager_.image(image)
                self.consecutive_fails_ = 0
                fails_in_collection = 0
                n_im_in_collection += 1
                yield image
            except MaxImagesReached:
                raise
            except StopIteration:
                break
            except Exception:
                fails_in_collection += 1
                _logger.exception(
                    '_scroll_collection: bad image: {0}'.format(image))
                self._failed_download()
            if fails_in_collection == self.max_fails_in_collection_:
                _logger.error('Too many bad images in collection {0}:  '
                              '{1} bad images.'.format(
                                  collection['id'], fails_in_collection))
                return

        _logger.info(
            'On neurovault.org: '
            '{0} image{1} matched query in collection {2}'.format(
                (n_im_in_collection if n_im_in_collection else 'no'),
                ('s' if n_im_in_collection > 1 else ''), collection['id']))

    @_managed_method()
    def _scroll_explicit(self):
        """Iterate over explicitely listed collections and images.

        Yields
        ------
        image : dict
            Metadata for an image.

        collection : dict
            Metadata for the image's collection.

        Raises
        ------

        RuntimeError
            If more than ``self.max_consecutive_fails_`` images (from
            the specified collections) have failed in a row.

        URLError, ValueError, errors raised by the download manager
            If an image specified in the `wanted_image_ids` cannot be
            downloaded.

        """
        collection_urls = [
            urljoin(_NEUROVAULT_COLLECTIONS_URL, str(col_id)) for
            col_id in self.wanted_collection_ids_ or [] if
            col_id not in self.visited_collections_]

        if(collection_urls):
            _logger.debug('Reading server neurovault data.')

        for collection in _yield_from_url_list(collection_urls):
            collection = self.download_manager_.collection(collection)
            for image in self._scroll_collection(collection):
                self.visited_images_.add(image['id'])
                yield image, collection

        image_urls = [urljoin(_NEUROVAULT_IMAGES_URL, str(im_id)) for
                      im_id in self.wanted_image_ids_ or [] if
                      im_id not in self.visited_images_]
        for image in _yield_from_url_list(image_urls):
            self.download_manager_.image(image)
            collection = _json_add_collection_dir(os.path.join(
                os.path.dirname(image['absolute_path']),
                'collection_metadata.json'))
            yield image, collection

    @_managed_method()
    def _scroll_filtered(self):
        """Iterate over collections matching the specified filters.

        Yields
        ------
        image : dict
            Metadata for an image.

        collection : dict
            Metadata for the image's collection.

        Raises
        ------
        MaxImagesReached
            If enough images have been downloaded.

        RuntimeError
            If more than ``self.max_consecutive_fails_`` images have
            failed in a row.

        """
        _logger.debug('Reading server neurovault data.')

        collections = _scroll_server_results(
            _NEUROVAULT_COLLECTIONS_URL, query_terms=self.collection_terms_,
            local_filter=self.collection_filter_,
            prefix_msg='Scroll collections: ', batch_size=self.batch_size_)

        while True:
            collection = None
            try:
                collection = next(collections)
                collection = self.download_manager_.collection(collection)
                good_collection = True
            except MaxImagesReached:
                raise
            except StopIteration:
                break
            except Exception:
                _logger.exception('scroll: bad collection: {0}'.format(
                    collection))
                self._failed_download()
                good_collection = False

            collection_content = self._scroll_collection(collection)
            while good_collection:
                try:
                    image = next(collection_content)
                except MaxImagesReached:
                    raise
                except StopIteration:
                    break
                yield image, collection

    @_managed_method()
    def _scroll_local(self):
        _logger.debug('Reading local neurovault data.')
        print('Reading local neurovault data.')
        collections = glob(
            os.path.join(
                self.neurovault_dir_, '*', 'collection_metadata.json'))

        for collection in filter(
                self.local_collection_filter_,
                map(_json_add_collection_dir, collections)):

            self.visited_collections_.add(collection['id'])
            images = glob(os.path.join(
                collection['absolute_path'], 'image_*_metadata.json'))

            for image in filter(self.local_image_filter_,
                                map(_json_add_im_files_paths, images)):
                if len(self.visited_images_) == self.max_images_:
                    return
                self.visited_images_.add(image['id'])
                image, collection = self.download_manager_.update(
                    image, collection)
                yield image, collection

        self.collection_filter_ = ResultFilter(
            {'id': NotIn(*self.visited_collections_)}).AND(
                self.collection_filter_)

        self.image_filter_ = ResultFilter(
            {'id': NotIn(*self.visited_images_)}).AND(
                self.image_filter_)

        found = len(self.visited_images_)
        print('{0} image{1} found on local disk.'.format(
            ('No' if not found else found), ('s' if found > 1 else '')))
        _logger.debug('{0} image{1} found on local disk.'.format(
            ('No' if not found else found), ('s' if found > 1 else '')))

    def scroll(self):
        """Iterate over neurovault.org content.

        Yields
        ------
        image : dict
            Metadata for an image.

        collection : dict
            Metadata for the image's collection.

        Raises
        ------
        MaxImagesReached
            If enough images have been downloaded.

        RuntimeError
            If more than ``self.max_consecutive_fails_`` images have
            failed in a row.

        URLError, ValueError, errors raised by the download manager
            If an image specified in the `wanted_image_ids` cannot be
            downloaded.

        """
        scroll_modes = {'filtered': self._scroll_filtered,
                        'explicit': self._scroll_explicit}

        if self.download_mode_ != 'overwrite':
            for image, collection in self._scroll_local():
                yield image, collection

        if self.download_mode_ == 'offline':
            return
        if (self.max_images_ is not None and
            len(self.visited_images_) >= self.max_images_):
            return

        server_data = scroll_modes[self.scroll_mode_]()
        while True:
            try:
                image, collection = next(server_data)
            except StopIteration:
                return
            except Exception:
                _logger.exception(
                    'Downloading data from server stopped early.')
                warnings.warn(
                    'Downloading data from Neurovault stopped early.')
                return
            yield image, collection


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


def _fetch_neurovault_impl(
        max_images=100,
        collection_terms=basic_collection_terms(),
        collection_filter=_empty_filter,
        image_terms=basic_image_terms(),
        image_filter=_empty_filter,
        collection_ids=None, image_ids=None,
        mode='download_new', data_dir=None,
        fetch_neurosynth_words=False, fetch_reduced_rep=False,
        download_manager=None, vectorize_words=True, **kwarg_image_filters):
    """Download data from neurovault.org and neurosynth.org."""

    if collection_ids is not None or image_ids is not None:
        max_images = None
    if max_images == 100:
        _logger.info(
            'fetch_neurovault: using default value of 100 for max_images. '
            'Set max_images to another value or None if you want more images.')
    collection_terms = dict(collection_terms)
    image_terms = dict(image_terms, **kwarg_image_filters)
    image_terms, collection_terms = _move_col_id(image_terms, collection_terms)

    neurovault_data_dir = _get_dataset_dir('neurovault', data_dir)
    if mode != 'offline' and not os.access(neurovault_data_dir, os.W_OK):
        warnings.warn("You don't have write access to neurovault dir: {0}; "
                      "fetch_neurovault is working offline.".format(
                          neurovault_data_dir))
        mode = 'offline'

    if download_manager is None and mode != 'offline':
        download_manager = DownloadManager(
            max_images=max_images,
            neurovault_data_dir=neurovault_data_dir,
            fetch_neurosynth_words=fetch_neurosynth_words,
            fetch_reduced_rep=fetch_reduced_rep)

    scroller = _DataScroller(neurovault_data_dir, mode,
                             collection_terms, collection_filter,
                             image_terms, image_filter,
                             collection_ids, image_ids,
                             download_manager, max_images).scroll()

    scroller = list(scroller)
    if not scroller:
        return None
    images_meta, collections_meta = map(list, zip(*scroller))
    images = [im_meta.get('absolute_path') for im_meta in images_meta]
    result = Bunch(images=images,
                   images_meta=images_meta,
                   collections_meta=collections_meta,
                   description=_get_dataset_descr('neurovault'))
    if fetch_neurosynth_words and vectorize_words:
        (result['word_frequencies'],
         result['vocabulary']) = neurosynth_words_vectorized(
             [meta.get('neurosynth_words_absolute_path') for
              meta in images_meta])
    return result


def fetch_neurovault(
        max_images=100,
        collection_terms=basic_collection_terms(),
        collection_filter=_empty_filter,
        image_terms=basic_image_terms(),
        image_filter=_empty_filter,
        mode='download_new', data_dir=None,
        fetch_neurosynth_words=False, fetch_reduced_rep=False,
        download_manager=None, vectorize_words=True, **kwarg_image_filters):
    """Download data from neurovault.org and neurosynth.org using filters.

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
        Maximum number of images to fetch. Ignored if `collection_ids`
        or `image_ids` is used.

    collection_terms : dict, optional (default=basic_collection_terms())
        Key, value pairs used to filter collection
        metadata. Collections for which
        ``collection_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be discarded.
        Ignored if `collection_ids` or `image_ids` is used.
        See documentation for ``basic_collection_terms`` for a
        description of the default selection criteria.

    collection_filter : Callable, optional (default=_empty_filter)
        Collections for which `collection_filter(collection_metadata)`
        is ``False`` will be discarded. Ignored if `collection_ids` or
        `image_ids` is used.

    image_terms : dict, optional (default=basic_image_terms())
        Key, value pairs used to filter image metadata. Images for
        which ``image_metadata['key'] == value`` is not ``True`` for
        every key, value pair will be discarded. Ignored if
        `collection_ids` or `image_ids` is used.
        See documentation for ``basic_image_terms`` for a
        description of the default selection criteria.

    image_filter : Callable, optional (default=_empty_filter)
        Images for which `image_filter(image_metadata)` is ``False``
        will be discarded. Ignored if `collection_ids` or `image_ids`
        is used.

    mode : {'download_new', 'overwrite', 'offline'}
        When to fetch an image from the server rather than the local
        disk.

        - 'download_new' (the default) means download only files that
          are not already on disk (regardless of modify date).
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    neurovault_data_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. Another
        directory may be used if the one that was specified is not
        valid.

    fetch_neurosynth_words : bool, optional (default=False)
        Wether to collect words from Neurosynth.

    fetch_reduced_rep : bool, optional (default=False)
        Wether to collect subsampled representations of images
        available on Neurovault.

    download_manager : BaseDownloadManager, optional (default=None)
        The download manager used to handle data from neurovault.org.

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

        If `fetch_neurosynth_words` was set, it also contains:

            - 'vocabulary', a list of words
            - 'word_frequencies', the weight of the words returned by
              neurosynth.org for each image, such that the weight of word
              `vocabulary[j]` for the image found in `images[i]` is
              `word_frequencies[i, j]`

    Notes
    -----
    Images and collections from disk are fetched before remote data.

    Some helpers are provided in the ``neurovault`` module to express
    filtering criteria in a less verbose manner:

        ``ResultFilter``, ``IsNull``, ``NotNull``, ``NotEqual``,
        ``GreaterOrEqual``, ``GreaterThan``, ``LessOrEqual``,
        ``LessThan``, ``IsIn``, ``NotIn``, ``Contains``,
        ``NotContains``, ``Pattern``.

    If you pass a single value to match against the collection id
    (wether as the 'id' field of the collection metadata or as the
    'collection_id' field of the image metadata), the server is
    directly queried for that collection, so
    ``fetch_neurovault(collection_id=40)`` is as efficient as
    ``fetch_neurovault(collection_ids=[40])`` (but in the former
    version the other filters will still be applied). This is not true
    for the image ids. If you pass a single value to match against any
    of the fields listed in ``_COL_FILTERS_AVAILABLE_ON_SERVER``,
    i.e., 'DOI', 'name', and 'owner_name', these filters can be
    applied by the server, limiting the amount of metadata we have to
    download: filtering on those fields makes the fetching faster
    because the filtering takes place on the server side.

    In `download_new` mode, if a file exists on disk, it is not
    downloaded again, even if the version on the server is newer. Use
    `overwrite` mode to force a new download (you can filter on the
    field ``modify_date`` to re-download the files that are newer on
    the server - see Examples section).

    Tries to yield `max_images` images; stops early if we have fetched
    all the images matching the filters or if an uncaught exception is
    raised during download

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
            max_images=None, mode='overwrite', modify_date=GreaterThan(newest))

    """
    return _fetch_neurovault_impl(
        max_images=max_images, collection_terms=collection_terms,
        collection_filter=collection_filter, image_terms=image_terms,
        image_filter=image_filter, mode=mode,
        data_dir=data_dir,
        fetch_neurosynth_words=fetch_neurosynth_words,
        fetch_reduced_rep=fetch_reduced_rep,
        download_manager=download_manager,
        vectorize_words=vectorize_words, **kwarg_image_filters)


def fetch_neurovault_ids(
        collection_ids=(), image_ids=(),
        mode='download_new', data_dir=None,
        fetch_neurosynth_words=False, fetch_reduced_rep=False,
        download_manager=None, vectorize_words=True):
    """Download images and collections from neurovault.org and neurosynth.org.

    Any downloaded data is saved on the local disk and subsequent
    calls to this function will first look for the data locally before
    querying the server for more if necessary.

    This is the fast way to get the data from the server if we already
    know which images or collections we want.

    Parameters
    ----------

    collection_ids : Container, optional (default=None)
        If not ``None``, ignore filters and terms and just download
        all collections from this list (and if specified all the
        images from `image_ids`)

    image_ids : Container, optional (default=None)
        If not ``None``, ignore filters and terms and just download
        all images from this list (and if specified all the
        collections from `collection_ids`)

    mode : {'download_new', 'overwrite', 'offline'}
        When to fetch an image from the server rather than the local
        disk.

        - 'download_new' (the default) means download only files that
          are not already on disk (regardless of modify date).
        - 'overwrite' means ignore files on disk and overwrite them.
        - 'offline' means load only data from disk; don't query server.

    neurovault_data_dir : str, optional (default=None)
        The directory we want to use for Neurovault data. Another
        directory may be used if the one that was specified is not
        valid.

    fetch_neurosynth_words : bool, optional (default=False)
        Wether to collect words from Neurosynth.

    fetch_reduced_rep : bool, optional (default=False)
        Wether to collect subsampled representations of images
        available on Neurovault.

    download_manager : BaseDownloadManager, optional (default=None)
        The download manager used to handle data from neurovault.org.

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

        If `fetch_neurosynth_words` was set, it also contains:

            - 'vocabulary', a list of words
            - 'word_frequencies', the weight of the words returned by
              neurosynth.org for each image, such that the weight of word
              `vocabulary[j]` for the image found in `images[i]` is
              `word_frequencies[i, j]`


    Raises
    ------
    RuntimeError
        If one of the explicitely required images or collections
        cannot be fetched.

    Notes
    -----
    Images and collections from disk are fetched before remote data.

    In `download_new` mode, if a file exists on disk, it is not
    downloaded again, even if the version on the server is newer. Use
    `overwrite` mode to force a new download (you can filter on the
    field ``modify_date`` to re-download the files that are newer on
    the server - see Examples section).

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
    return _fetch_neurovault_impl(
        mode=mode,
        collection_ids=collection_ids, image_ids=image_ids,
        data_dir=data_dir,
        fetch_neurosynth_words=fetch_neurosynth_words,
        fetch_reduced_rep=fetch_reduced_rep,
        download_manager=download_manager,
        vectorize_words=vectorize_words)
