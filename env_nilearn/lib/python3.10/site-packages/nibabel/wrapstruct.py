# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Class to wrap numpy structured array

============
 wrapstruct
============

The :class:`WrapStruct` class is a wrapper around a numpy structured array
type.

It implements:

* Mappingness from the underlying structured array fields
* ``from_fileobj``, ``write_to`` methods to read and write data to fileobj
* A mechanism for setting checks and fixes to the data on object creation
* Endianness guessing, and on-the-fly swapping

The :class:`LabeledWrapStruct` subclass adds:

* A pretty printing mechanism whereby field values can be displayed as
  corresponding strings (see :meth:`LabeledWrapStruct.get_value_label` and
  :meth:`LabeledWrapStruct.__str_`)

Mappingness
-----------

You can access and set fields of the contained structarr using standard
__getitem__ / __setitem__ syntax:

    wrapped['field'] = 10

Wrapped structures also implement general mappingness:

    wrapped.keys()
    wrapped.items()
    wrapped.values()

Properties::

    .endianness (read only)
    .binaryblock (read only)
    .structarr (read only)

Methods::

    .as_byteswapped(endianness)
    .check_fix()
    .__str__
    .__eq__
    .__ne__
    .get_value_label(name)

Class methods::

    .diagnose_binaryblock
    .as_byteswapped(endianness)
    .write_to(fileobj)
    .from_fileobj(fileobj)
    .default_structarr() - return default structured array
    .guessed_endian(structarr) - return guessed endian code from this structarr

Class variables:
    template_dtype - native endian version of dtype for contained structarr

Consistency checks
------------------

We have a file, and we would like information as to whether there are any
problems with the binary data in this file, and whether they are fixable.
``WrapStruct`` can hold checks for internal consistency of the contained data::

   wrapped = WrapStruct.from_fileobj(open('myfile.bin'), check=False)
   dx_result = WrapStruct.diagnose_binaryblock(wrapped.binaryblock)

This will run all known checks, with no fixes, returning a string with
diagnostic output. See below for the ``check=False`` flag.

In creating a ``WrapStruct`` object, we often want to check the consistency of
the contained data.  The checks can test for problems of various levels of
severity.  If the problem is severe enough, it should raise an Error.  So, with
data that is consistent - no error::

   wrapped = WrapStruct.from_fileobj(good_fileobj)

whereas::

   wrapped = WrapStruct.from_fileobj(bad_fileobj)

would raise some error, with output to logging (see below).

If we want the created object, come what may::

   hdr = WrapStruct.from_fileobj(bad_fileobj, check=False)

We set the error level (the level of problem that the ``check=True``
versions will accept as OK) from global defaults::

   import nibabel as nib
   nib.imageglobals.error_level = 30

The same for logging::

   nib.imageglobals.logger = logger
"""

from __future__ import annotations

import numpy as np

from . import imageglobals as imageglobals
from .batteryrunners import BatteryRunner
from .volumeutils import Recoder, endian_codes, native_code, pretty_mapping, swapped_code


class WrapStructError(Exception):
    pass


class WrapStruct:
    # placeholder datatype
    template_dtype = np.dtype([('integer', 'i2')])

    def __init__(self, binaryblock=None, endianness=None, check=True):
        """Initialize WrapStruct from binary data block

        Parameters
        ----------
        binaryblock : {None, string} optional
            binary block to set into object.  By default, None, in
            which case we insert the default empty block
        endianness : {None, '<','>', other endian code} string, optional
            endianness of the binaryblock.  If None, guess endianness
            from the data.
        check : bool, optional
            Whether to check content of binary data in initialization.
            Default is True.

        Examples
        --------
        >>> wstr1 = WrapStruct() # a default structure
        >>> wstr1.endianness == native_code
        True
        >>> wstr1['integer']
        array(0, dtype=int16)
        >>> wstr1['integer'] = 1
        >>> wstr1['integer']
        array(1, dtype=int16)
        """
        if binaryblock is None:
            self._structarr = self.__class__.default_structarr(endianness)
            return
        # check size
        if len(binaryblock) != self.template_dtype.itemsize:
            raise WrapStructError('Binary block is wrong size')
        wstr = np.ndarray(shape=(), dtype=self.template_dtype, buffer=binaryblock)
        if endianness is None:
            endianness = self.__class__.guessed_endian(wstr)
        else:
            endianness = endian_codes[endianness]
        if endianness != native_code:
            dt = self.template_dtype.newbyteorder(endianness)
            wstr = np.ndarray(shape=(), dtype=dt, buffer=binaryblock)
        self._structarr = wstr.copy()
        if check:
            self.check_fix()

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        """Return read structure with given or guessed endiancode

        Parameters
        ----------
        fileobj : file-like object
           Needs to implement ``read`` method
        endianness : None or endian code, optional
           Code specifying endianness of read data

        Returns
        -------
        wstr : WrapStruct object
           WrapStruct object initialized from data in fileobj
        """
        raw_str = fileobj.read(klass.template_dtype.itemsize)
        return klass(raw_str, endianness, check)

    @property
    def binaryblock(self):
        """binary block of data as string

        Returns
        -------
        binaryblock : string
            string giving binary data block

        Examples
        --------
        >>> # Make default empty structure
        >>> wstr = WrapStruct()
        >>> len(wstr.binaryblock)
        2
        """
        return self._structarr.tobytes()

    def write_to(self, fileobj):
        """Write structure to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method

        Returns
        -------
        None

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> from io import BytesIO
        >>> str_io = BytesIO()
        >>> wstr.write_to(str_io)
        >>> wstr.binaryblock == str_io.getvalue()
        True
        """
        fileobj.write(self.binaryblock)

    @property
    def endianness(self):
        """endian code of binary data

        The endianness code gives the current byte order
        interpretation of the binary data.

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> code = wstr.endianness
        >>> code == native_code
        True

        Notes
        -----
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization, or occasionally byteswapping the
        data - but this is done via the as_byteswapped method
        """
        if self._structarr.dtype.isnative:
            return native_code
        return swapped_code

    def copy(self):
        """Return copy of structure

        >>> wstr = WrapStruct()
        >>> wstr['integer'] = 3
        >>> wstr2 = wstr.copy()
        >>> wstr2 is wstr
        False
        >>> wstr2['integer']
        array(3, dtype=int16)
        """
        return self.__class__(self.binaryblock, self.endianness, check=False)

    def __eq__(self, other):
        """equality between two structures defined by binaryblock

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr2 = WrapStruct()
        >>> wstr == wstr2
        True
        >>> wstr3 = WrapStruct(endianness=swapped_code)
        >>> wstr == wstr3
        True
        """
        this_end = self.endianness
        this_bb = self.binaryblock
        try:
            other_end = other.endianness
            other_bb = other.binaryblock
        except AttributeError:
            return False
        if this_end == other_end:
            return this_bb == other_bb
        other_bb = other._structarr.byteswap().tobytes()
        return this_bb == other_bb

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, item):
        """Return values from structure data

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr['integer'] == 0
        True
        """
        return self._structarr[item]

    def __setitem__(self, item, value):
        """Set values in structured data

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr['integer'] = 3
        >>> wstr['integer']
        array(3, dtype=int16)
        """
        self._structarr[item] = value

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        """Return keys from structured data"""
        return list(self.template_dtype.names)

    def values(self):
        """Return values from structured data"""
        data = self._structarr
        return [data[key] for key in self.template_dtype.names]

    def items(self):
        """Return items from structured data"""
        return zip(self.keys(), self.values())

    def get(self, k, d=None):
        """Return value for the key k if present or d otherwise"""
        return self._structarr[k] if k in self.keys() else d

    def check_fix(self, logger=None, error_level=None):
        """Check structured data with checks

        Parameters
        ----------
        logger : None or logging.Logger
        error_level : None or int
            Level of error severity at which to raise error.  Any error of
            severity >= `error_level` will cause an exception.
        """
        if logger is None:
            logger = imageglobals.logger
        if error_level is None:
            error_level = imageglobals.error_level
        battrun = BatteryRunner(self.__class__._get_checks())
        self, reports = battrun.check_fix(self)
        for report in reports:
            report.log_raise(logger, error_level)

    @classmethod
    def diagnose_binaryblock(klass, binaryblock, endianness=None):
        """Run checks over binary data, return string"""
        wstr = klass(binaryblock, endianness=endianness, check=False)
        battrun = BatteryRunner(klass._get_checks())
        reports = battrun.check_only(wstr)
        return '\n'.join([report.message for report in reports if report.message])

    @classmethod
    def guessed_endian(self, mapping):
        """Guess intended endianness from mapping-like ``mapping``

        Parameters
        ----------
        wstr : mapping-like
            Something implementing a mapping.  We will guess the endianness
            from looking at the field values

        Returns
        -------
        endianness : {'<', '>'}
           Guessed endianness of binary data in ``wstr``
        """
        raise NotImplementedError

    @classmethod
    def default_structarr(klass, endianness=None):
        """Return structured array for default structure with given endianness"""
        dt = klass.template_dtype
        if endianness is not None:
            endianness = endian_codes[endianness]
            dt = dt.newbyteorder(endianness)
        return np.zeros((), dtype=dt)

    @property
    def structarr(self):
        """Structured data, with data fields

        Examples
        --------
        >>> wstr1 = WrapStruct() # with default data
        >>> an_int = wstr1.structarr['integer']
        >>> wstr1.structarr = None
        Traceback (most recent call last):
           ...
        AttributeError: ...
        """
        return self._structarr

    def __str__(self):
        """Return string representation for printing"""
        summary = f"{self.__class__} object, endian='{self.endianness}'"
        return '\n'.join([summary, pretty_mapping(self)])

    def as_byteswapped(self, endianness=None):
        """return new byteswapped object with given ``endianness``

        Guaranteed to make a copy even if endianness is the same as
        the current endianness.

        Parameters
        ----------
        endianness : None or string, optional
           endian code to which to swap.  None means swap from current
           endianness, and is the default

        Returns
        -------
        wstr : ``WrapStruct``
           ``WrapStruct`` object with given endianness

        Examples
        --------
        >>> wstr = WrapStruct()
        >>> wstr.endianness == native_code
        True
        >>> bs_wstr = wstr.as_byteswapped()
        >>> bs_wstr.endianness == swapped_code
        True
        >>> bs_wstr = wstr.as_byteswapped(swapped_code)
        >>> bs_wstr.endianness == swapped_code
        True
        >>> bs_wstr is wstr
        False
        >>> bs_wstr == wstr
        True

        If you write to the resulting byteswapped data, it does not
        change the original.

        >>> bs_wstr['integer'] = 3
        >>> bs_wstr == wstr
        False

        If you swap to the same endianness, it returns a copy

        >>> nbs_wstr = wstr.as_byteswapped(native_code)
        >>> nbs_wstr.endianness == native_code
        True
        >>> nbs_wstr is wstr
        False
        """
        current = self.endianness
        if endianness is None:
            if current == native_code:
                endianness = swapped_code
            else:
                endianness = native_code
        else:
            endianness = endian_codes[endianness]
        if endianness == current:
            return self.copy()
        wstr_data = self._structarr.byteswap()
        return self.__class__(wstr_data.tobytes(), endianness, check=False)

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class"""
        return ()


class LabeledWrapStruct(WrapStruct):
    """A WrapStruct with some fields having value labels for printing etc"""

    _field_recoders: dict[str, Recoder] = {}  # for recoding values for str

    def get_value_label(self, fieldname):
        """Returns label for coded field

        A coded field is an int field containing codes that stand for
        discrete values that also have string labels.

        Parameters
        ----------
        fieldname : str
           name of header field to get label for

        Returns
        -------
        label : str
           label for code value in header field `fieldname`

        Raises
        ------
        ValueError
            if field is not coded.

        Examples
        --------
        >>> from nibabel.volumeutils import Recoder
        >>> recoder = Recoder(((1, 'one'), (2, 'two')), ('code', 'label'))
        >>> class C(LabeledWrapStruct):
        ...     template_dtype = np.dtype([('datatype', 'i2')])
        ...     _field_recoders = dict(datatype = recoder)
        >>> hdr  = C()
        >>> hdr.get_value_label('datatype')
        '<unknown code 0>'
        >>> hdr['datatype'] = 2
        >>> hdr.get_value_label('datatype')
        'two'
        """
        if fieldname not in self._field_recoders:
            raise ValueError(f'{fieldname} not a coded field')
        code = int(self._structarr[fieldname])
        try:
            return self._field_recoders[fieldname].label[code]
        except KeyError:
            return f'<unknown code {code}>'

    def __str__(self):
        """Return string representation for printing"""
        summary = f"{self.__class__} object, endian='{self.endianness}'"

        def _getter(obj, key):
            try:
                return obj.get_value_label(key)
            except ValueError:
                return obj[key]

        return '\n'.join([summary, pretty_mapping(self, _getter)])
