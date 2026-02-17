# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Utility functions for analyze-like formats"""

from __future__ import annotations

import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext

import numpy as np

from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet

if ty.TYPE_CHECKING:
    import io

    import numpy.typing as npt

    Scalar = np.number | float

    K = ty.TypeVar('K')
    V = ty.TypeVar('V')
    DT = ty.TypeVar('DT', bound=np.generic)

sys_is_le = sys.byteorder == 'little'
native_code = sys_is_le and '<' or '>'
swapped_code = sys_is_le and '>' or '<'

_endian_codes = (  # numpy code, aliases
    ('<', 'little', 'l', 'le', 'L', 'LE'),
    ('>', 'big', 'BIG', 'b', 'be', 'B', 'BE'),
    (native_code, 'native', 'n', 'N', '=', '|', 'i', 'I'),
    (swapped_code, 'swapped', 's', 'S', '!'),
)
# We'll put these into the Recoder class after we define it

#: default compression level when writing gz and bz2 files
default_compresslevel = 1


class Recoder:
    """class to return canonical code(s) from code or aliases

    The concept is a lot easier to read in the implementation and
    tests than it is to explain, so...

    >>> # If you have some codes, and several aliases, like this:
    >>> code1 = 1; aliases1=['one', 'first']
    >>> code2 = 2; aliases2=['two', 'second']
    >>> # You might want to do this:
    >>> codes = [[code1]+aliases1,[code2]+aliases2]
    >>> recodes = Recoder(codes)
    >>> recodes.code['one']
    1
    >>> recodes.code['second']
    2
    >>> recodes.code[2]
    2
    >>> # Or maybe you have a code, a label and some aliases
    >>> codes=((1,'label1','one', 'first'),(2,'label2','two'))
    >>> # you might want to get back the code or the label
    >>> recodes = Recoder(codes, fields=('code','label'))
    >>> recodes.code['first']
    1
    >>> recodes.code['label1']
    1
    >>> recodes.label[2]
    'label2'
    >>> # For convenience, you can get the first entered name by
    >>> # indexing the object directly
    >>> recodes[2]
    2
    """

    fields: tuple[str, ...]

    def __init__(
        self,
        codes: ty.Sequence[ty.Sequence[ty.Hashable]],
        fields: ty.Sequence[str] = ('code',),
        map_maker: type[ty.Mapping[ty.Hashable, ty.Hashable]] = dict,
    ):
        """Create recoder object

        ``codes`` give a sequence of code, alias sequences
        ``fields`` are names by which the entries in these sequences can be
        accessed.

        By default ``fields`` gives the first column the name
        "code".  The first column is the vector of first entries
        in each of the sequences found in ``codes``.  Thence you can
        get the equivalent first column value with ob.code[value],
        where value can be a first column value, or a value in any of
        the other columns in that sequence.

        You can give other columns names too, and access them in the
        same way - see the examples in the class docstring.

        Parameters
        ----------
        codes : sequence of sequences
            Each sequence defines values (codes) that are equivalent
        fields : {('code',) string sequence}, optional
            names by which elements in sequences can be accessed
        map_maker: callable, optional
            constructor for dict-like objects used to store key value pairs.
            Default is ``dict``.  ``map_maker()`` generates an empty mapping.
            The mapping need only implement ``__getitem__, __setitem__, keys,
            values``.
        """
        self.fields = tuple(fields)
        self.field1 = {}  # a placeholder for the check below
        for name in fields:
            if name in self.__dict__:
                raise KeyError(f'Input name {name} already in object dict')
            self.__dict__[name] = map_maker()
        self.field1 = self.__dict__[fields[0]]
        self.add_codes(codes)

    def __getattr__(self, key: str) -> ty.Mapping[ty.Hashable, ty.Hashable]:
        # By setting this, we let static analyzers know that dynamic attributes will
        # be dict-like (Mapping).
        # However, __getattr__ is called if looking up the field in __dict__ fails,
        # so we only get here if the attribute is really missing.
        raise AttributeError(f'{self.__class__.__name__!r} object has no attribute {key!r}')

    def add_codes(self, code_syn_seqs: ty.Sequence[ty.Sequence[ty.Hashable]]) -> None:
        """Add codes to object

        Parameters
        ----------
        code_syn_seqs : sequence
            sequence of sequences, where each sequence ``S = code_syn_seqs[n]``
            for n in 0..len(code_syn_seqs), is a sequence giving values in the
            same order as ``self.fields``.  Each S should be at least of the
            same length as ``self.fields``.  After this call, if ``self.fields
            == ['field1', 'field2'], then ``self.field1[S[n]] == S[0]`` for all
            n in 0..len(S) and ``self.field2[S[n]] == S[1]`` for all n in
            0..len(S).

        Examples
        --------
        >>> code_syn_seqs = ((2, 'two'), (1, 'one'))
        >>> rc = Recoder(code_syn_seqs)
        >>> rc.value_set() == set((1,2))
        True
        >>> rc.add_codes(((3, 'three'), (1, 'first')))
        >>> rc.value_set() == set((1,2,3))
        True
        >>> print(rc.value_set())  # set is actually ordered
        OrderedSet([2, 1, 3])
        """
        for code_syns in code_syn_seqs:
            # Add all the aliases
            for alias in code_syns:
                # For all defined fields, make every value in the sequence be
                # an entry to return matching index value.
                for field_ind, field_name in enumerate(self.fields):
                    self.__dict__[field_name][alias] = code_syns[field_ind]

    def __getitem__(self, key: ty.Hashable) -> ty.Hashable:
        """Return value from field1 dictionary (first column of values)

        Returns same value as ``obj.field1[key]`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code[key]``

        >>> codes = ((1, 'one'), (2, 'two'))
        >>> Recoder(codes)['two']
        2
        """
        return self.field1[key]

    def __contains__(self, key: ty.Hashable) -> bool:
        """True if field1 in recoder contains `key`"""
        return key in self.field1

    def keys(self):
        """Return all available code and alias values

        Returns same value as ``obj.field1.keys()`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code.keys()``

        >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
        >>> k = Recoder(codes).keys()
        >>> set(k) == set([1, 2, 'one', 'repeat value', 'two'])
        True
        """
        return self.field1.keys()

    def value_set(self, name: str | None = None) -> OrderedSet:
        """Return OrderedSet of possible returned values for column

        By default, the column is the first column.

        Returns same values as ``set(obj.field1.values())`` and,
        with the default initializing``fields`` argument of
        fields=('code',), this will return the same as
        ``set(obj.code.values())``

        Parameters
        ----------
        name : {None, string}
            Where default of none gives result for first column

        >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
        >>> vs = Recoder(codes).value_set()
        >>> vs == set([1, 2]) # Sets are not ordered, hence this test
        True
        >>> rc = Recoder(codes, fields=('code', 'label'))
        >>> rc.value_set('label') == set(('one', 'two', 'repeat value'))
        True
        """
        if name is None:
            d = self.field1
        else:
            d = self.__dict__[name]
        return OrderedSet(d.values())


# Endian code aliases
endian_codes = Recoder(_endian_codes)


class DtypeMapper(dict[ty.Hashable, ty.Hashable]):
    """Specialized mapper for numpy dtypes

    We pass this mapper into the Recoder class to deal with numpy dtype
    hashing.

    The hashing problem is that dtypes that compare equal may not have the same
    hash.  This is true for numpys up to the current at time of writing
    (1.6.0).  For numpy 1.2.1 at least, even dtypes that look exactly the same
    in terms of fields don't always have the same hash.  This makes dtypes
    difficult to use as keys in a dictionary.

    This class wraps a dictionary in order to implement a __getitem__ to deal
    with dtype hashing. If the key doesn't appear to be in the mapping, and it
    is a dtype, we compare (using ==) all known dtype keys to the input key,
    and return any matching values for the matching key.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dtype_keys: list[np.dtype] = []

    def __setitem__(self, key: ty.Hashable, value: ty.Hashable) -> None:
        """Set item into mapping, checking for dtype keys

        Cache dtype keys for comparison test in __getitem__
        """
        super().__setitem__(key, value)
        if isinstance(key, np.dtype):
            self._dtype_keys.append(key)

    def __getitem__(self, key: ty.Hashable) -> ty.Hashable:
        """Get item from mapping, checking for dtype keys

        First do simple hash lookup, then check for a dtype key that has failed
        the hash lookup.  Look then for any known dtype keys that compare equal
        to `key`.
        """
        try:
            return super().__getitem__(key)
        except KeyError:
            pass
        if isinstance(key, np.dtype):
            for dt in self._dtype_keys:
                if key == dt:
                    return super().__getitem__(dt)
        raise KeyError(key)


def pretty_mapping(
    mapping: ty.Mapping[K, V],
    getterfunc: ty.Callable[[ty.Mapping[K, V], K], V] | None = None,
) -> str:
    """Make pretty string from mapping

    Adjusts text column to print values on basis of longest key.
    Probably only sensible if keys are mainly strings.

    You can pass in a callable that does clever things to get the values
    out of the mapping, given the names.  By default, we just use
    ``__getitem__``

    Parameters
    ----------
    mapping : mapping
       implementing iterator returning keys and .items()
    getterfunc : None or callable
       callable taking two arguments, ``obj`` and ``key`` where ``obj``
       is the passed mapping.  If None, just use ``lambda obj, key:
       obj[key]``

    Returns
    -------
    str : string

    Examples
    --------
    >>> d = {'a key': 'a value'}
    >>> print(pretty_mapping(d))
    a key  : a value
    >>> class C: # to control ordering, show get_ method
    ...     def __iter__(self):
    ...         return iter(('short_field','longer_field'))
    ...     def __getitem__(self, key):
    ...         if key == 'short_field':
    ...             return 0
    ...         if key == 'longer_field':
    ...             return 'str'
    ...     def get_longer_field(self):
    ...         return 'method string'
    >>> def getter(obj, key):
    ...     # Look for any 'get_<name>' methods
    ...     try:
    ...         return obj.__getattribute__('get_' + key)()
    ...     except AttributeError:
    ...         return obj[key]
    >>> print(pretty_mapping(C(), getter))
    short_field   : 0
    longer_field  : method string
    """
    if getterfunc is None:
        getterfunc = getitem
    mxlen = max(len(str(name)) for name in mapping)
    fmt = '%%-%ds  : %%s' % mxlen
    out = []
    for name in mapping:
        value = getterfunc(mapping, name)
        out.append(fmt % (name, value))
    return '\n'.join(out)


def make_dt_codes(codes_seqs: ty.Sequence[ty.Sequence]) -> Recoder:
    """Create full dt codes Recoder instance from datatype codes

    Include created numpy dtype (from numpy type) and opposite endian
    numpy dtype

    Parameters
    ----------
    codes_seqs : sequence of sequences
       contained sequences make be length 3 or 4, but must all be the same
       length. Elements are data type code, data type name, and numpy
       type (such as ``np.float32``).  The fourth element is the nifti string
       representation of the code (e.g. "NIFTI_TYPE_FLOAT32")

    Returns
    -------
    rec : ``Recoder`` instance
       Recoder that, by default, returns ``code`` when indexed with any
       of the corresponding code, name, type, dtype, or swapped dtype.
       You can also index with ``niistring`` values if codes_seqs had sequences
       of length 4 instead of 3.
    """
    fields = ['code', 'label', 'type']
    len0 = len(codes_seqs[0])
    if len0 not in (3, 4):
        raise ValueError('Sequences must be length 3 or 4')
    if len0 == 4:
        fields.append('niistring')
    dt_codes = []
    for seq in codes_seqs:
        if len(seq) != len0:
            raise ValueError('Sequences must all have the same length')
        np_type = seq[2]
        this_dt = np.dtype(np_type)
        # Add swapped dtype to synonyms
        code_syns = list(seq) + [this_dt, this_dt.newbyteorder(swapped_code)]
        dt_codes.append(code_syns)
    return Recoder(dt_codes, fields + ['dtype', 'sw_dtype'], DtypeMapper)


def _is_compressed_fobj(fobj: io.IOBase) -> bool:
    """Return True if fobj represents a compressed data file-like object"""
    return isinstance(fobj, COMPRESSED_FILE_LIKES)


def array_from_file(
    shape: tuple[int, ...],
    in_dtype: np.dtype[DT],
    infile: io.IOBase,
    offset: int = 0,
    order: ty.Literal['C', 'F'] = 'F',
    mmap: bool | ty.Literal['c', 'r', 'r+'] = True,
) -> npt.NDArray[DT]:
    """Get array from file with specified shape, dtype and file offset

    Parameters
    ----------
    shape : sequence
        sequence specifying output array shape
    in_dtype : numpy dtype
        fully specified numpy dtype, including correct endianness
    infile : file-like
        open file-like object implementing at least read() and seek()
    offset : int, optional
        offset in bytes into `infile` to start reading array data. Default is 0
    order : {'F', 'C'} string
        order in which to write data.  Default is 'F' (fortran order).
    mmap : {True, False, 'c', 'r', 'r+'}
        `mmap` controls the use of numpy memory mapping for reading data.  If
        False, do not try numpy ``memmap`` for data array.  If one of {'c',
        'r', 'r+'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
        True gives the same behavior as ``mmap='c'``.  If `infile` cannot be
        memory-mapped, ignore `mmap` value and read array from file.

    Returns
    -------
    arr : array-like
        array like object that can be sliced, containing data

    Examples
    --------
    >>> from io import BytesIO
    >>> bio = BytesIO()
    >>> arr = np.arange(6).reshape(1,2,3)
    >>> _ = bio.write(arr.tobytes('F'))  # outputs int
    >>> arr2 = array_from_file((1,2,3), arr.dtype, bio)
    >>> np.all(arr == arr2)
    True
    >>> bio = BytesIO()
    >>> _ = bio.write(b' ' * 10)
    >>> _ = bio.write(arr.tobytes('F'))
    >>> arr2 = array_from_file((1,2,3), arr.dtype, bio, 10)
    >>> np.all(arr == arr2)
    True
    """
    if mmap not in (True, False, 'c', 'r', 'r+'):
        raise ValueError("mmap value should be one of True, False, 'c', 'r', 'r+'")
    in_dtype = np.dtype(in_dtype)
    # Get file-like object from Opener instance
    infile = getattr(infile, 'fobj', infile)
    if mmap and not _is_compressed_fobj(infile):
        mode = 'c' if mmap is True else mmap
        try:  # Try memmapping file on disk
            return np.memmap(infile, in_dtype, mode=mode, shape=shape, order=order, offset=offset)
            # The error raised by memmap, for different file types, has
            # changed in different incarnations of the numpy routine
        except (AttributeError, TypeError, ValueError):
            pass
    if len(shape) == 0:
        return np.array([], in_dtype)
    # Use reduce and mul to work around numpy integer overflow
    n_bytes = reduce(mul, shape) * in_dtype.itemsize
    if n_bytes == 0:
        return np.array([], in_dtype)
    # Read data from file
    infile.seek(offset)
    if hasattr(infile, 'readinto'):
        data_bytes = bytearray(n_bytes)
        n_read = infile.readinto(data_bytes)
        needs_copy = False
    else:
        data_bytes = infile.read(n_bytes)
        n_read = len(data_bytes)
        needs_copy = True
    if n_bytes != n_read:
        raise OSError(
            f'Expected {n_bytes} bytes, got {n_read} bytes from '
            f"{getattr(infile, 'name', 'object')}\n - could the file be damaged?"
        )
    arr: np.ndarray = np.ndarray(shape, in_dtype, buffer=data_bytes, order=order)
    if needs_copy:
        return arr.copy()
    arr.flags.writeable = True
    return arr


def array_to_file(
    data: npt.ArrayLike,
    fileobj: io.IOBase,
    out_dtype: np.dtype | None = None,
    offset: int = 0,
    intercept: Scalar = 0.0,
    divslope: Scalar | None = 1.0,
    mn: Scalar | None = None,
    mx: Scalar | None = None,
    order: ty.Literal['C', 'F'] = 'F',
    nan2zero: bool = True,
) -> None:
    """Helper function for writing arrays to file objects

    Writes arrays as scaled by `intercept` and `divslope`, and clipped
    at (prescaling) `mn` minimum, and `mx` maximum.

    * Clip `data` array at min `mn`, max `max` where there are not None ->
      ``clipped`` (this is *pre scale clipping*)
    * Scale ``clipped`` with ``clipped_scaled = (clipped - intercept) /
      divslope``
    * Clip ``clipped_scaled`` to fit into range of `out_dtype` (*post scale
      clipping*) -> ``clipped_scaled_clipped``
    * If converting to integer `out_dtype` and `nan2zero` is True, set NaN
      values in ``clipped_scaled_clipped`` to 0
    * Write ``clipped_scaled_clipped_n2z`` to fileobj `fileobj` starting at
      offset `offset` in memory layout `order`

    Parameters
    ----------
    data : array-like
        array or array-like to write.
    fileobj : file-like
        file-like object implementing ``write`` method.
    out_dtype : None or dtype, optional
        dtype to write array as.  Data array will be coerced to this dtype
        before writing. If None (default) then use input data type.
    offset : None or int, optional
        offset into fileobj at which to start writing data. Default is 0. None
        means start at current file position
    intercept : scalar, optional
        scalar to subtract from data, before dividing by ``divslope``.  Default
        is 0.0
    divslope : None or scalar, optional
        scalefactor to *divide* data by before writing.  Default is 1.0. If
        None, there is no valid data, we write zeros.
    mn : scalar, optional
        minimum threshold in (unscaled) data, such that all data below this
        value are set to this value. Default is None (no threshold). The
        typical use is to set -np.inf in the data to have this value (which
        might be the minimum non-finite value in the data).
    mx : scalar, optional
        maximum threshold in (unscaled) data, such that all data above this
        value are set to this value. Default is None (no threshold). The
        typical use is to set np.inf in the data to have this value (which
        might be the maximum non-finite value in the data).
    order : {'F', 'C'}, optional
        memory order to write array.  Default is 'F'
    nan2zero : {True, False}, optional
        Whether to set NaN values to 0 when writing integer output.  Defaults
        to True.  If False, NaNs will be represented as numpy does when
        casting; this depends on the underlying C library and is undefined. In
        practice `nan2zero` == False might be a good choice when you completely
        sure there will be no NaNs in the data. This value ignored for float
        output types.  NaNs are treated as zero *before* applying `intercept`
        and `divslope` - so an array ``[np.nan]`` with an `intercept` of 10
        becomes ``[-10]`` after conversion to integer `out_dtype` with
        `nan2zero` set.  That is because you will likely apply `divslope` and
        `intercept` in reverse order when reading the data back, returning the
        zero you probably expected from the input NaN.

    Examples
    --------
    >>> from io import BytesIO
    >>> sio = BytesIO()
    >>> data = np.arange(10, dtype=np.float64)
    >>> array_to_file(data, sio, np.float64)
    >>> sio.getvalue() == data.tobytes('F')
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)  # outputs 0
    >>> array_to_file(data, sio, np.int16)
    >>> sio.getvalue() == data.astype(np.int16).tobytes()
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)
    >>> array_to_file(data.byteswap(), sio, np.float64)
    >>> sio.getvalue() == data.byteswap().tobytes('F')
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)
    >>> array_to_file(data, sio, np.float64, order='C')
    >>> sio.getvalue() == data.tobytes('C')
    True
    """
    # Shield special case
    if not np.isfinite(np.array((intercept, 1.0 if divslope is None else divslope))).all():
        raise ValueError('divslope and intercept must be finite')
    if divslope == 0:
        raise ValueError('divslope cannot be zero')
    data = np.asanyarray(data)
    in_dtype = data.dtype
    if out_dtype is None:
        out_dtype = in_dtype
    else:
        out_dtype = np.dtype(out_dtype)
    if offset is not None:
        seek_tell(fileobj, offset)
    if divslope is None or (mn, mx) == (0, 0) or ((mn is not None and mx is not None) and mx < mn):
        write_zeros(fileobj, data.size * out_dtype.itemsize)
        return
    if order not in 'FC':
        raise ValueError('Order should be one of F or C')
    # Simple cases
    pre_clips = None if (mn is None and mx is None) else (mn, mx)
    null_scaling = intercept == 0 and divslope == 1
    if in_dtype.type == np.void:
        if not null_scaling:
            raise ValueError('Cannot scale non-numeric types')
        if pre_clips is not None:
            raise ValueError('Cannot clip non-numeric types')
        return _write_data(data, fileobj, out_dtype, order)
    if pre_clips is not None:
        pre_clips = _dt_min_max(in_dtype, *pre_clips)
    if null_scaling and np.can_cast(in_dtype, out_dtype):
        return _write_data(data, fileobj, out_dtype, order, pre_clips=pre_clips)
    # Force upcasting for floats by making atleast_1d.
    slope, inter = (np.atleast_1d(v) for v in (divslope, intercept))
    # Default working point type for applying slope / inter
    if slope.dtype.kind in 'iu':
        slope = slope.astype(float)
    if inter.dtype.kind in 'iu':
        inter = inter.astype(float)
    in_kind = in_dtype.kind
    out_kind = out_dtype.kind
    if out_kind in 'fc':
        return _write_data(
            data, fileobj, out_dtype, order, slope=slope, inter=inter, pre_clips=pre_clips
        )
    assert out_kind in 'iu'
    if in_kind in 'iu':
        if null_scaling:
            # Must be large int to small int conversion; add clipping to
            # pre scale thresholds
            mn, mx = _dt_min_max(in_dtype, mn, mx)
            mn_out, mx_out = _dt_min_max(out_dtype)
            pre_clips = max(mn, mn_out), min(mx, mx_out)  # type: ignore[type-var]
            return _write_data(data, fileobj, out_dtype, order, pre_clips=pre_clips)
        # In any case, we do not want to check for nans because we've already
        # disallowed scaling that generates nans
        nan2zero = False
    # We are either scaling into c/floats or starting with c/floats, then we're
    # going to integers
    # Because we're going to integers, complex inter and slope will only slow
    # us down, cast to float
    slope, inter = (v.astype(_matching_float(v.dtype)) for v in (slope, inter))
    # We'll do the thresholding on the scaled data, so turn off the
    # thresholding on the unscaled data
    pre_clips = None
    # We may need to cast the original array to another type
    cast_in_dtype = in_dtype
    if in_kind == 'c':
        # Cast to floats before anything else
        cast_in_dtype = np.dtype(_matching_float(in_dtype))
    elif in_kind == 'f' and in_dtype.itemsize == 2:
        # Make sure we don't use float16 as a working type
        cast_in_dtype = np.dtype(np.float32)
    w_type = working_type(cast_in_dtype, slope, inter)
    dt_mnmx = _dt_min_max(cast_in_dtype, mn, mx)
    # We explore for a good precision to avoid infs and clipping
    # Find smallest float type equal or larger than the current working
    # type, that can contain range of extremes after scaling, without going
    # to +-inf
    extremes = np.array(dt_mnmx, dtype=cast_in_dtype)
    w_type = best_write_scale_ftype(extremes, slope, inter, w_type)
    # Push up precision by casting the slope, inter
    slope, inter = (v.astype(w_type) for v in (slope, inter))
    # We need to know the result of applying slope and inter to the min and
    # max of the array, in order to clip the output array, after applying
    # the slope and inter.  Otherwise we'd need to clip twice, once before
    # applying (slope, inter), and again after, to ensure we have not hit
    # over- or under-flow. For the same reason we need to know the result of
    # applying slope, inter to 0, in order to fill in the nan output value
    # after scaling etc. We could fill with 0 before scaling, but then we'd
    # have to do an extra copy before filling nans with 0, to avoid
    # overwriting the input array
    # Run min, max, 0 through scaling / rint
    specials = np.array(dt_mnmx + (0,), dtype=w_type)
    if inter != 0.0:
        specials = specials - inter
    if slope != 1.0:
        specials = specials / slope
    assert specials.dtype.type == w_type
    post_mn, post_mx, nan_fill = np.rint(specials)
    if post_mn > post_mx:  # slope could be negative
        post_mn, post_mx = post_mx, post_mn
    # Make sure that the thresholds exclude any value that will get badly cast
    # to the integer type.  This is not the same as using the maximumum of the
    # output dtype as thresholds, because these may not be exactly represented
    # in the float type.
    #
    # The thresholds assume that the data are in `wtype` dtype after applying
    # the slope and intercept.
    both_mn, both_mx = shared_range(w_type, out_dtype)
    # Check that nan2zero output value is in range
    if nan2zero and not both_mn <= nan_fill <= both_mx:
        # Estimated error for (0 - inter) / slope is 2 * eps * abs(inter /
        # slope).  Assume errors are for working float type. Round for integer
        # rounding
        est_err = np.round(2 * np.finfo(w_type).eps * abs(inter / slope))
        if (nan_fill < both_mn and abs(nan_fill - both_mn) < est_err) or (
            nan_fill > both_mx and abs(nan_fill - both_mx) < est_err
        ):
            # nan_fill can be (just) outside clip range
            nan_fill = np.clip(nan_fill, both_mn, both_mx)
        else:
            raise ValueError(
                f'nan_fill == {nan_fill}, outside safe int range '
                f'({int(both_mn)}-{int(both_mx)}); '
                'change scaling or set nan2zero=False?'
            )
    # Make sure non-nan output clipped to shared range
    post_mn = np.max([post_mn, both_mn])
    post_mx = np.min([post_mx, both_mx])
    in_cast = None if cast_in_dtype == in_dtype else cast_in_dtype
    return _write_data(
        data,
        fileobj,
        out_dtype,
        order,
        in_cast=in_cast,
        pre_clips=pre_clips,
        inter=inter,
        slope=slope,
        post_clips=(post_mn, post_mx),
        nan_fill=nan_fill if nan2zero else None,
    )


def _write_data(
    data: np.ndarray,
    fileobj: io.IOBase,
    out_dtype: np.dtype,
    order: ty.Literal['C', 'F'],
    in_cast: np.dtype | None = None,
    pre_clips: tuple[Scalar | None, Scalar | None] | None = None,
    inter: Scalar | np.ndarray = 0.0,
    slope: Scalar | np.ndarray = 1.0,
    post_clips: tuple[Scalar | None, Scalar | None] | None = None,
    nan_fill: Scalar | None = None,
) -> None:
    """Write array `data` to `fileobj` as `out_dtype` type, layout `order`

    Does not modify `data` in-place.

    Parameters
    ----------
    data : ndarray
    fileobj : object
        implementing ``obj.write``
    out_dtype : numpy type
        Type to which to cast output data just before writing
    order : {'F', 'C'}
        memory layout of array in fileobj after writing
    in_cast : None or numpy type, optional
        If not None, initial cast to do on `data` slices before further
        processing
    pre_clips : None or 2-sequence, optional
        If not None, minimum and maximum of input values at which to clip.
    inter : scalar or array, optional
        Intercept to subtract before writing ``out = data - inter``
    slope : scalar or array, optional
        Slope by which to divide before writing ``out2 = out / slope``
    post_clips : None or 2-sequence, optional
        If not None, minimum and maximum of scaled values at which to clip.
    nan_fill : None or scalar, optional
        If not None, values that were NaN in `data` will receive `nan_fill`
        in array as output to disk (after scaling).
    """
    data = np.squeeze(data)
    if data.ndim < 2:  # Trick to allow loop over rows for 1D arrays
        data = np.atleast_2d(data)
    elif order == 'F':
        data = data.T
    nan_need_copy = (pre_clips, in_cast, inter, slope, post_clips) == (None, None, 0, 1, None)
    for dslice in data:  # cycle over first dimension to save memory
        if pre_clips is not None:
            dslice = np.clip(dslice, *pre_clips)
        if in_cast is not None:
            dslice = dslice.astype(in_cast)
        if inter != 0.0:
            dslice = dslice - inter
        if slope != 1.0:
            dslice = dslice / slope
        if post_clips is not None:
            dslice = np.clip(np.rint(dslice), *post_clips)
        if nan_fill is not None:
            nans = np.isnan(dslice)
            if np.any(nans):
                if nan_need_copy:
                    dslice = dslice.copy()
                dslice[nans] = nan_fill
        if dslice.dtype != out_dtype:
            dslice = dslice.astype(out_dtype)
        fileobj.write(dslice.tobytes())


def _dt_min_max(
    dtype_like: npt.DTypeLike, mn: Scalar | None = None, mx: Scalar | None = None
) -> tuple[Scalar, Scalar]:
    dt = np.dtype(dtype_like)
    if dt.kind in 'fc':
        dt_mn, dt_mx = (-np.inf, np.inf)
    elif dt.kind in 'iu':
        info = np.iinfo(dt)
        dt_mn, dt_mx = (info.min, info.max)
    else:
        raise ValueError('unknown dtype')
    return dt_mn if mn is None else mn, dt_mx if mx is None else mx


_CSIZE2FLOAT: dict[int, type[np.floating]] = {
    8: np.float32,
    16: np.float64,
    24: np.longdouble,
    32: np.longdouble,
}


def _matching_float(np_type: npt.DTypeLike) -> type[np.floating]:
    """Return floating point type matching `np_type`"""
    dtype = np.dtype(np_type)
    if dtype.kind not in 'cf':
        raise ValueError('Expecting float or complex type as input')
    if issubclass(dtype.type, np.floating):
        return dtype.type
    return _CSIZE2FLOAT[dtype.itemsize]


def write_zeros(fileobj: io.IOBase, count: int, block_size: int = 8194) -> None:
    """Write `count` zero bytes to `fileobj`

    Parameters
    ----------
    fileobj : file-like object
        with ``write`` method
    count : int
        number of bytes to write
    block_size : int, optional
        largest continuous block to write.
    """
    nblocks = int(count // block_size)
    rem = count % block_size
    blk = b'\x00' * block_size
    for bno in range(nblocks):
        fileobj.write(blk)
    fileobj.write(b'\x00' * rem)


def seek_tell(fileobj: io.IOBase, offset: int, write0: bool = False) -> None:
    """Seek in `fileobj` or check we're in the right place already

    Parameters
    ----------
    fileobj : file-like
        object implementing ``seek`` and (if seek raises an OSError) ``tell``
    offset : int
        position in file to which to seek
    write0 : {False, True}, optional
        If True, and standard seek fails, try to write zeros to the file to
        reach `offset`.  This can be useful when writing bz2 files, that cannot
        do write seeks.
    """
    try:
        fileobj.seek(offset)
    except OSError as e:
        # This can be a negative seek in write mode for gz file object or any
        # seek in write mode for a bz2 file object
        pos = fileobj.tell()
        if pos == offset:
            return
        if not write0:
            raise OSError(str(e))
        if pos > offset:
            raise OSError("Can't write to seek backwards")
        fileobj.write(b'\x00' * (offset - pos))
        assert fileobj.tell() == offset


def apply_read_scaling(
    arr: np.ndarray,
    slope: Scalar | None = None,
    inter: Scalar | None = None,
) -> np.ndarray:
    """Apply scaling in `slope` and `inter` to array `arr`

    This is for loading the array from a file (as opposed to the reverse
    scaling when saving an array to file)

    Return data will be ``arr * slope + inter``. The trick is that we have to
    find a good precision to use for applying the scaling.  The heuristic is
    that the data is always upcast to the higher of the types from `arr,
    `slope`, `inter` if `slope` and / or `inter` are not default values. If the
    dtype of `arr` is an integer, then we assume the data more or less fills
    the integer range, and upcast to a type such that the min, max of
    ``arr.dtype`` * scale + inter, will be finite.

    Parameters
    ----------
    arr : array-like
    slope : None or float, optional
        slope value to apply to `arr` (``arr * slope + inter``).  None
        corresponds to a value of 1.0
    inter : None or float, optional
        intercept value to apply to `arr` (``arr * slope + inter``).  None
        corresponds to a value of 0.0

    Returns
    -------
    ret : array
        array with scaling applied.  Maybe upcast in order to give room for the
        scaling. If scaling is default (1, 0), then `ret` may be `arr` ``ret is
        arr``.
    """
    if slope is None:
        slope = 1.0
    if inter is None:
        inter = 0.0
    if (slope, inter) == (1, 0):
        return arr
    shape = arr.shape
    # Force float / float upcasting by promoting to arrays
    slope1d, inter1d = (np.atleast_1d(v) for v in (slope, inter))
    arr = np.atleast_1d(arr)
    if arr.dtype.kind in 'iu':
        # int to float; get enough precision to avoid infs
        # Find floating point type for which scaling does not overflow,
        # starting at given type
        default = slope1d.dtype.type if slope1d.dtype.kind == 'f' else np.float64
        ftype = int_scinter_ftype(arr.dtype, slope1d, inter1d, default)
        slope1d = slope1d.astype(ftype)
        inter1d = inter1d.astype(ftype)
    if slope1d != 1.0:
        arr = arr * slope1d
    if inter1d != 0.0:
        arr = arr + inter1d
    return arr.reshape(shape)


def working_type(
    in_type: npt.DTypeLike,
    slope: npt.ArrayLike = 1.0,
    inter: npt.ArrayLike = 0.0,
) -> type[np.number]:
    """Return array type from applying `slope`, `inter` to array of `in_type`

    Numpy type that results from an array of type `in_type` being combined with
    `slope` and `inter`. It returns something like the dtype type of
    ``((np.zeros((2,), dtype=in_type) - inter) / slope)``, but ignoring the
    actual values of `slope` and `inter`.

    Note that you would not necessarily get the same type by applying slope and
    inter the other way round.  Also, you'll see that the order in which slope
    and inter are applied is the opposite of the order in which they are
    passed.

    Parameters
    ----------
    in_type : numpy type specifier
        Numpy type of input array.  Any valid input for ``np.dtype()``
    slope : scalar, optional
        slope to apply to array.  If 1.0 (default), ignore this value and its
        type.
    inter : scalar, optional
        intercept to apply to array.  If 0.0 (default), ignore this value and
        its type.

    Returns
    -------
    wtype: numpy type
        Numpy type resulting from applying `inter` and `slope` to array of type
        `in_type`.
    """
    val = np.array([1], dtype=in_type)
    # Don't use real values to avoid overflows.  Promote to 1D to avoid scalar
    # casting rules.  Don't use ones_like, zeros_like because of a bug in numpy
    # <= 1.5.1 in converting complex192 / complex256 scalars.
    if inter != 0:
        val = val + np.array([0], dtype=np.array(inter).dtype)
    if slope != 1:
        val = val / np.array([1], dtype=np.array(slope).dtype)
    return val.dtype.type


def int_scinter_ftype(
    ifmt: type[np.integer],
    slope: npt.ArrayLike = 1.0,
    inter: npt.ArrayLike = 0.0,
    default: type[np.floating] = np.float32,
) -> type[np.floating]:
    """float type containing int type `ifmt` * `slope` + `inter`

    Return float type that can represent the max and the min of the `ifmt` type
    after multiplication with `slope` and addition of `inter` with something
    like ``np.array([imin, imax], dtype=ifmt) * slope + inter``.

    Note that ``slope`` and ``inter`` get promoted to 1D arrays for this
    purpose to avoid the numpy scalar casting rules, which prevent scalars
    upcasting the array.

    Parameters
    ----------
    ifmt : object
        numpy integer type (e.g. np.int32)
    slope : float, optional
        slope, default 1.0
    inter : float, optional
        intercept, default 0.0
    default_out : object, optional
        numpy floating point type, default is ``np.float32``

    Returns
    -------
    ftype : object
        numpy floating point type

    Examples
    --------
    >>> int_scinter_ftype(np.int8, 1.0, 0.0) == np.float32
    True
    >>> int_scinter_ftype(np.int8, 1e38, 0.0) == np.float64
    True

    Notes
    -----
    It is difficult to make floats overflow with just addition because the
    deltas are so large at the extremes of floating point.  For example::

        >>> arr = np.array([np.finfo(np.float32).max], dtype=np.float32)
        >>> res = arr + np.iinfo(np.int16).max
        >>> arr == res
        array([ True])
    """
    ii = np.iinfo(ifmt)
    tst_arr = np.array([ii.min, ii.max], dtype=ifmt)
    try:
        return _ftype4scaled_finite(tst_arr, slope, inter, 'read', default)
    except ValueError:
        raise ValueError('Overflow using highest floating point type')


def best_write_scale_ftype(
    arr: np.ndarray,
    slope: npt.ArrayLike = 1.0,
    inter: npt.ArrayLike = 0.0,
    default: type[np.number] = np.float32,
) -> type[np.floating]:
    """Smallest float type to contain range of ``arr`` after scaling

    Scaling that will be applied to ``arr`` is ``(arr - inter) / slope``.

    Note that ``slope`` and ``inter`` get promoted to 1D arrays for this
    purpose to avoid the numpy scalar casting rules, which prevent scalars
    upcasting the array.

    Parameters
    ----------
    arr : array-like
        array that will be scaled
    slope : array-like, optional
        scalar such that output array will be ``(arr - inter) / slope``.
    inter : array-like, optional
        scalar such that output array will be ``(arr - inter) / slope``
    default : numpy type, optional
        minimum float type to return

    Returns
    -------
    ftype : numpy type
        Best floating point type for scaling.  If no floating point type
        prevents overflow, return the top floating point type.  If the input
        array ``arr`` already contains inf values, return the greater of the
        input type and the default type.

    Examples
    --------
    >>> arr = np.array([0, 1, 2], dtype=np.int16)
    >>> best_write_scale_ftype(arr, 1, 0) is np.float32
    True

    Specify higher default return value

    >>> best_write_scale_ftype(arr, 1, 0, default=np.float64) is np.float64
    True

    Even large values that don't overflow don't change output

    >>> arr = np.array([0, np.finfo(np.float32).max], dtype=np.float32)
    >>> best_write_scale_ftype(arr, 1, 0) is np.float32
    True

    Scaling > 1 reduces output values, so no upcast needed

    >>> best_write_scale_ftype(arr, np.float32(2), 0) is np.float32
    True

    Scaling < 1 increases values, so upcast may be needed (and is here)

    >>> best_write_scale_ftype(arr, np.float32(0.5), 0) is np.float64
    True
    """
    default = better_float_of(arr.dtype.type, default)
    if not np.all(np.isfinite(arr)):
        return default
    try:
        return _ftype4scaled_finite(arr, slope, inter, 'write', default)
    except ValueError:
        return OK_FLOATS[-1]


def better_float_of(
    first: npt.DTypeLike,
    second: npt.DTypeLike,
    default: type[np.floating] = np.float32,
) -> type[np.floating]:
    """Return more capable float type of `first` and `second`

    Return `default` if neither of `first` or `second` is a float

    Parameters
    ----------
    first : numpy type specifier
        Any valid input to `np.dtype()``
    second : numpy type specifier
        Any valid input to `np.dtype()``
    default : numpy type specifier, optional
        Any valid input to `np.dtype()``

    Returns
    -------
    better_type : numpy type
        More capable of `first` or `second` if both are floats; if only one is
        a float return that, otherwise return `default`.

    Examples
    --------
    >>> better_float_of(np.float32, np.float64) is np.float64
    True
    >>> better_float_of(np.float32, 'i4') is np.float32
    True
    >>> better_float_of('i2', 'u4') is np.float32
    True
    >>> better_float_of('i2', 'u4', np.float64) is np.float64
    True
    """
    first = np.dtype(first)
    second = np.dtype(second)
    default = np.dtype(default).type
    if issubclass(first.type, np.floating):
        if issubclass(second.type, np.floating) and first.itemsize < second.itemsize:
            return second.type
        return first.type
    if issubclass(second.type, np.floating):
        return second.type
    return default


def _ftype4scaled_finite(
    tst_arr: np.ndarray,
    slope: npt.ArrayLike,
    inter: npt.ArrayLike,
    direction: ty.Literal['read', 'write'] = 'read',
    default: type[np.floating] = np.float32,
) -> type[np.floating]:
    """Smallest float type for scaling of `tst_arr` that does not overflow"""
    assert direction in ('read', 'write')
    if default not in OK_FLOATS and default is np.longdouble:
        # Omitted longdouble
        return default
    def_ind = OK_FLOATS.index(default)
    # promote to arrays to avoid numpy scalar casting rules
    tst_arr = np.atleast_1d(tst_arr)
    slope = np.atleast_1d(slope)
    inter = np.atleast_1d(inter)
    for ftype in OK_FLOATS[def_ind:]:
        tst_trans = tst_arr.copy()
        slope = slope.astype(ftype)
        inter = inter.astype(ftype)
        try:
            with warnings.catch_warnings():
                # Error on overflows to short circuit the logic
                warnings.filterwarnings('error', '.*overflow.*', RuntimeWarning)
                if direction == 'read':  # as in reading of image from disk
                    if slope != 1.0:
                        tst_trans = tst_trans * slope
                    if inter != 0.0:
                        tst_trans = tst_trans + inter
                elif direction == 'write':
                    if inter != 0.0:
                        tst_trans = tst_trans - inter
                    if slope != 1.0:
                        tst_trans = tst_trans / slope
            # Double-check that result is finite
            if np.all(np.isfinite(tst_trans)):
                return ftype
        except RuntimeWarning:
            pass
    raise ValueError('Overflow using highest floating point type')


@ty.overload
def finite_range(
    arr: npt.ArrayLike, check_nan: ty.Literal[False] = False
) -> tuple[Scalar, Scalar]: ...


@ty.overload
def finite_range(
    arr: npt.ArrayLike, check_nan: ty.Literal[True]
) -> tuple[Scalar, Scalar, bool]: ...


def finite_range(
    arr: npt.ArrayLike,
    check_nan: bool = False,
) -> tuple[Scalar, Scalar, bool] | tuple[Scalar, Scalar]:
    """Get range (min, max) or range and flag (min, max, has_nan) from `arr`

    Parameters
    ----------
    arr : array-like
    check_nan : {False, True}, optional
        Whether to return third output, a bool signaling whether there are NaN
        values in `arr`

    Returns
    -------
    mn : scalar
       minimum of values in (flattened) array
    mx : scalar
       maximum of values in (flattened) array
    has_nan : bool
       Returned if `check_nan` is True. `has_nan` is True if there are one or
       more NaN values in `arr`

    Examples
    --------
    >>> a = np.array([[-1, 0, 1],[np.inf, np.nan, -np.inf]])
    >>> finite_range(a)
    (-1.0, 1.0)
    >>> a = np.array([[-1, 0, 1],[np.inf, np.nan, -np.inf]])
    >>> finite_range(a, check_nan=True)
    (-1.0, 1.0, True)
    >>> a = np.array([[np.nan],[np.nan]])
    >>> finite_range(a) == (np.inf, -np.inf)
    True
    >>> a = np.array([[-3, 0, 1],[2,-1,4]], dtype=int)
    >>> finite_range(a)
    (-3, 4)
    >>> a = np.array([[1, 0, 1],[2,3,4]], dtype=np.uint)
    >>> finite_range(a)
    (0, 4)
    >>> a = a + 1j
    >>> finite_range(a)
    (1j, (4+1j))
    >>> a = np.zeros((2,), dtype=[('f1', 'i2')])
    >>> finite_range(a)
    Traceback (most recent call last):
       ...
    TypeError: Can only handle numeric types
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        if check_nan:
            return (np.inf, -np.inf, False)
        return (np.inf, -np.inf)
    # Resort array to slowest->fastest memory change indices
    stride_order = np.argsort(arr.strides)[::-1]
    sarr = arr.transpose(stride_order)
    kind = sarr.dtype.kind
    if kind in 'iu':
        if check_nan:
            return np.min(sarr), np.max(sarr), False
        return np.min(sarr), np.max(sarr)
    if kind not in 'cf':
        raise TypeError('Can only handle numeric types')
    # Deal with 1D arrays in loop below
    sarr = np.atleast_2d(sarr)
    # Loop to avoid big temporary arrays
    has_nan = False
    n_slices = sarr.shape[0]
    maxes = np.zeros(n_slices, dtype=sarr.dtype) - np.inf
    mins = np.zeros(n_slices, dtype=sarr.dtype) + np.inf
    for s in range(n_slices):
        this_slice = sarr[s]  # view
        if not has_nan:
            maxes[s] = np.max(this_slice)
            # May have a non-nan non-inf max before we trip on min. If so,
            # record so we don't recalculate
            max_good = False
            if np.isnan(maxes[s]):
                has_nan = True
            elif maxes[s] != np.inf:
                max_good = True
                mins[s] = np.min(this_slice)
                if mins[s] != -np.inf:
                    # Only case where we escape the default np.isfinite
                    # algorithm
                    continue
        tmp = this_slice[np.isfinite(this_slice)]
        if tmp.size == 0:  # No finite values
            # Reset max, min in case set in tests above
            maxes[s] = -np.inf
            mins[s] = np.inf
            continue
        if not max_good:
            maxes[s] = np.max(tmp)
        mins[s] = np.min(tmp)
    if check_nan:
        return np.nanmin(mins), np.nanmax(maxes), has_nan
    return np.nanmin(mins), np.nanmax(maxes)


def shape_zoom_affine(
    shape: ty.Sequence[int] | np.ndarray,
    zooms: ty.Sequence[float] | np.ndarray,
    x_flip: bool = True,
) -> np.ndarray:
    """Get affine implied by given shape and zooms

    We get the translations from the center of the image (implied by
    `shape`).

    Parameters
    ----------
    shape : (N,) array-like
       shape of image data. ``N`` is the number of dimensions
    zooms : (N,) array-like
       zooms (voxel sizes) of the image
    x_flip : {True, False}
       whether to flip the X row of the affine.  Corresponds to
       radiological storage on disk.

    Returns
    -------
    aff : (4,4) array
       affine giving correspondence of voxel coordinates to mm
       coordinates, taking the center of the image as origin

    Examples
    --------
    >>> shape = (3, 5, 7)
    >>> zooms = (3, 2, 1)
    >>> shape_zoom_affine((3, 5, 7), (3, 2, 1))
    array([[-3.,  0.,  0.,  3.],
           [ 0.,  2.,  0., -4.],
           [ 0.,  0.,  1., -3.],
           [ 0.,  0.,  0.,  1.]])
    >>> shape_zoom_affine((3, 5, 7), (3, 2, 1), False)
    array([[ 3.,  0.,  0., -3.],
           [ 0.,  2.,  0., -4.],
           [ 0.,  0.,  1., -3.],
           [ 0.,  0.,  0.,  1.]])
    """
    shape = np.asarray(shape)
    zooms = np.array(zooms)  # copy because of flip below
    ndims = len(shape)
    if ndims != len(zooms):
        raise ValueError('Should be same length of zooms and shape')
    if ndims >= 3:
        shape = shape[:3]
        zooms = zooms[:3]
    else:
        full_shape = np.ones((3,))
        full_zooms = np.ones((3,))
        full_shape[:ndims] = shape[:]
        full_zooms[:ndims] = zooms[:]
        shape = full_shape
        zooms = full_zooms
    if x_flip:
        zooms[0] *= -1
    # Get translations from center of image
    origin = (shape - 1) / 2.0
    aff = np.eye(4)
    aff[:3, :3] = np.diag(zooms)
    aff[:3, -1] = -origin * zooms
    return aff


def rec2dict(rec: np.ndarray) -> dict[str, np.generic | np.ndarray]:
    """Convert recarray to dictionary

    Also converts scalar values to scalars

    Parameters
    ----------
    rec : ndarray
       structured ndarray

    Returns
    -------
    dct : dict
       dict with key, value pairs as for `rec`

    Examples
    --------
    >>> r = np.zeros((), dtype = [('x', 'i4'), ('s', 'S10')])
    >>> d = rec2dict(r)
    >>> d == {'x': 0, 's': b''}
    True
    """
    dct = {}
    for key in rec.dtype.fields:
        val = rec[key]
        try:
            val = val.item()
        except ValueError:
            pass
        dct[key] = val
    return dct


def fname_ext_ul_case(fname: str) -> str:
    """`fname` with ext changed to upper / lower case if file exists

    Check for existence of `fname`.  If it does exist, return unmodified.  If
    it doesn't, check for existence of `fname` with case changed from lower to
    upper, or upper to lower.  Return this modified `fname` if it exists.
    Otherwise return `fname` unmodified

    Parameters
    ----------
    fname : str
        filename.

    Returns
    -------
    mod_fname : str
        filename, maybe with extension of opposite case
    """
    if exists(fname):
        return fname
    froot, ext = splitext(fname)
    if ext == ext.lower():
        mod_fname = froot + ext.upper()
        if exists(mod_fname):
            return mod_fname
    elif ext == ext.upper():
        mod_fname = froot + ext.lower()
        if exists(mod_fname):
            return mod_fname
    return fname
