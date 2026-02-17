# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test binary header objects

This is a root testing class, used in the Analyze and other tests as a
framework for all the tests common to the Analyze types

Refactoring TODO maybe
----------------------

binaryblock
diagnose_binaryblock

-> bytes, diagnose_bytes

With deprecation warnings

_field_recoders -> field_recoders
"""

import logging
from io import BytesIO, StringIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError

INTEGER_TYPES = sctypes['int'] + sctypes['uint']


def log_chk(hdr, level):
    """Utility method to check header checking / logging

    Asserts that log entry appears during ``hdr.check_fix`` for logging level
    below `level`.

    Parameters
    ----------
    hdr : instance
        Instance of header class, with methods ``copy`` and check_fix``.  The
        header has some minor error (defect) which can be detected with
        ``check_fix``.
    level : int
        Level (severity) of defect present in `hdr`.  When logging threshold is
        at or below `level`, a message appears in the default log (we test that
        happens).

    Returns
    -------
    hdrc : instance
        Header, with defect corrected.
    message : str
        Message generated in log when defect was detected.
    raiser : tuple
        Tuple of error type, callable, arguments that will raise an exception
        when then defect is detected.  Can be empty.  Check with ``if raiser !=
        (): assert_raises(*raiser)``.
    """
    str_io = StringIO()
    logger = logging.getLogger('test.logger')
    handler = logging.StreamHandler(str_io)
    logger.addHandler(handler)
    str_io.truncate(0)
    hdrc = hdr.copy()
    if level == 0:  # Should never log or raise error
        logger.setLevel(0)
        hdrc.check_fix(logger=logger, error_level=0)
        assert str_io.getvalue() == ''
        logger.removeHandler(handler)
        return hdrc, '', ()
    # Non zero defect level, test above and below threshold.
    # Set error level above defect level to prevent exception when defect
    # detected.
    e_lev = level + 1
    # Logging level above threshold, no log.
    logger.setLevel(level + 1)
    hdrc.check_fix(logger=logger, error_level=e_lev)
    assert str_io.getvalue() == ''
    # Logging level below threshold, log appears, store logged message
    logger.setLevel(level - 1)
    hdrc = hdr.copy()
    hdrc.check_fix(logger=logger, error_level=e_lev)
    assert str_io.getvalue() != ''
    message = str_io.getvalue().strip()
    logger.removeHandler(handler)
    # When error level == level, check_fix should raise an error
    hdrc2 = hdr.copy()
    raiser = (HeaderDataError, hdrc2.check_fix, logger, level)
    return hdrc, message, raiser


class _TestWrapStructBase:
    """Class implements base tests for binary headers

    It serves as a base class for other binary header tests
    """

    header_class = None

    def get_bad_bb(self):
        # Value for the binaryblock that will raise an error on checks. None
        # means do not check
        return None

    def test_general_init(self):
        hdr = self.header_class()
        # binaryblock has length given by header data dtype
        binblock = hdr.binaryblock
        assert len(binblock) == hdr.structarr.dtype.itemsize
        # Endianness will be native by default for empty header
        assert hdr.endianness == native_code
        # But you can change this if you want
        hdr = self.header_class(endianness='swapped')
        assert hdr.endianness == swapped_code
        # You can also pass in a check flag, without data this has no
        # effect
        hdr = self.header_class(check=False)

    def _set_something_into_hdr(self, hdr):
        # Called from test_bytes test method.  Specific to the header data type
        raise NotImplementedError('Not in base type')

    def test__eq__(self):
        # Test equal and not equal
        hdr1 = self.header_class()
        hdr2 = self.header_class()
        assert hdr1 == hdr2
        self._set_something_into_hdr(hdr1)
        assert hdr1 != hdr2
        self._set_something_into_hdr(hdr2)
        assert hdr1 == hdr2
        # Check byteswapping maintains equality
        hdr3 = hdr2.as_byteswapped()
        assert hdr2 == hdr3
        # Check comparing to funny thing says no
        assert hdr1 != None
        assert hdr1 != 1

    def test_to_from_fileobj(self):
        # Successful write using write_to
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        str_io.seek(0)
        hdr2 = self.header_class.from_fileobj(str_io)
        assert hdr2.endianness == native_code
        assert hdr2.binaryblock == hdr.binaryblock

    def test_mappingness(self):
        hdr = self.header_class()
        with pytest.raises(ValueError):
            hdr['nonexistent key'] = 0.1
        hdr_dt = hdr.structarr.dtype
        keys = hdr.keys()
        assert keys == list(hdr)
        vals = hdr.values()
        assert len(vals) == len(keys)
        assert keys == list(hdr_dt.names)
        for key, val in hdr.items():
            assert_array_equal(hdr[key], val)
        # verify that .get operates as destined
        assert hdr.get('nonexistent key') is None
        assert hdr.get('nonexistent key', 'default') == 'default'
        assert hdr.get(keys[0]) == vals[0]
        assert hdr.get(keys[0], 'default') == vals[0]

        # make sure .get returns values which evaluate to False. We have to
        # use a different falsy value depending on the data type of the first
        # header field.
        falsyval = 0 if np.issubdtype(hdr_dt[0], np.number) else b''

        hdr[keys[0]] = falsyval
        assert hdr[keys[0]] == falsyval
        assert hdr.get(keys[0]) == falsyval
        assert hdr.get(keys[0], -1) == falsyval

    def test_endianness_ro(self):
        # endianness is a read only property
        """Its use in initialization tested in the init tests.
        Endianness gives endian interpretation of binary data. It is
        read only because the only common use case is to set the
        endianness on initialization (or occasionally byteswapping the
        data) - but this is done via via the as_byteswapped method
        """
        hdr = self.header_class()
        with pytest.raises(AttributeError):
            hdr.endianness = '<'

    def test_endian_guess(self):
        # Check guesses of endian
        eh = self.header_class()
        assert eh.endianness == native_code
        hdr_data = eh.structarr.copy()
        hdr_data = hdr_data.byteswap(swapped_code)
        eh_swapped = self.header_class(hdr_data.tobytes())
        assert eh_swapped.endianness == swapped_code

    def test_binblock_is_file(self):
        # Checks that the binary string representation is the whole of the
        # header file.  This is true for Analyze types, but not true Nifti
        # single file headers, for example, because they will have extension
        # strings following.  More generally, there may be other perhaps
        # optional data after the binary block, in which case you will need to
        # override this test
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        assert str_io.getvalue() == hdr.binaryblock

    def test_structarr(self):
        # structarr attribute also read only
        hdr = self.header_class()
        # Just check we can get structarr
        hdr.structarr
        # That it's read only
        with pytest.raises(AttributeError):
            hdr.structarr = 0

    def log_chk(self, hdr, level):
        return log_chk(hdr, level)

    def assert_no_log_err(self, hdr):
        """Assert that no logging or errors result from this `hdr`"""
        fhdr, message, raiser = self.log_chk(hdr, 0)
        assert (fhdr, message) == (hdr, '')

    def test_bytes(self):
        # Test get of bytes
        hdr1 = self.header_class()
        bb = hdr1.binaryblock
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        # Do a set into the header, and try again.  The specifics of 'setting
        # something' will depend on the nature of the bytes object
        self._set_something_into_hdr(hdr1)
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        # Short and long binaryblocks give errors
        # (here set through init)
        with pytest.raises(WrapStructError):
            self.header_class(bb[:-1])
        with pytest.raises(WrapStructError):
            self.header_class(bb + b'\x00')
        # Checking set to true by default, and prevents nonsense being
        # set into the header.
        bb_bad = self.get_bad_bb()
        if bb_bad is None:
            return
        with imageglobals.LoggingOutputSuppressor():
            with pytest.raises(HeaderDataError):
                self.header_class(bb_bad)
        # now slips past without check
        _ = self.header_class(bb_bad, check=False)

    def test_as_byteswapped(self):
        # Check byte swapping
        hdr = self.header_class()
        assert hdr.endianness == native_code
        # same code just returns a copy
        hdr2 = hdr.as_byteswapped(native_code)
        assert not hdr is hdr2
        # Different code gives byteswapped copy
        hdr_bs = hdr.as_byteswapped(swapped_code)
        assert hdr_bs.endianness == swapped_code
        assert hdr.binaryblock != hdr_bs.binaryblock
        # Note that contents is not rechecked on swap / copy

        class DC(self.header_class):
            def check_fix(self, *args, **kwargs):
                raise Exception

        # Assumes check=True default
        with pytest.raises(Exception):
            DC(hdr.binaryblock)
        hdr = DC(hdr.binaryblock, check=False)
        hdr2 = hdr.as_byteswapped(native_code)
        hdr_bs = hdr.as_byteswapped(swapped_code)

    def test_empty_check(self):
        # Empty header should be error free
        hdr = self.header_class()
        hdr.check_fix(error_level=0)

    def _dxer(self, hdr):
        # Return diagnostics on bytes in `hdr`
        binblock = hdr.binaryblock
        return self.header_class.diagnose_binaryblock(binblock)

    def test_str(self):
        hdr = self.header_class()
        # Check something returns from str
        s1 = str(hdr)
        assert len(s1) > 0


class _TestLabeledWrapStruct(_TestWrapStructBase):
    """Test a wrapstruct with value labeling"""

    def test_get_value_label(self):
        # Test get value label method
        # Make a new class to avoid overwriting recoders of original
        class MyHdr(self.header_class):
            _field_recoders = {}

        hdr = MyHdr()
        # Key not existing raises error
        with pytest.raises(ValueError):
            hdr.get_value_label('improbable')
        # Even if there is a recoder
        assert 'improbable' not in hdr.keys()
        rec = Recoder([[0, 'fullness of heart']], ('code', 'label'))
        hdr._field_recoders['improbable'] = rec
        with pytest.raises(ValueError):
            hdr.get_value_label('improbable')
        # If the key exists in the structure, and is intable, then we can recode
        for key, value in hdr.items():
            # No recoder at first
            with pytest.raises(ValueError):
                hdr.get_value_label(0)
            if not value.dtype.type in INTEGER_TYPES or not np.isscalar(value):
                continue
            code = int(value)
            rec = Recoder([[code, 'fullness of heart']], ('code', 'label'))
            hdr._field_recoders[key] = rec
            assert hdr.get_value_label(key) == 'fullness of heart'
            # If key exists, but value is missing, we get 'unknown code'
            # Speculating that we can set code value 0 or 1
            new_code = 1 if code == 0 else 0
            hdr[key] = new_code
            assert hdr.get_value_label(key) == f'<unknown code {new_code}>'


class MyWrapStruct(WrapStruct):
    """An example wrapped struct class"""

    template_dtype = np.dtype([('an_integer', 'i2'), ('a_str', 'S10')])

    @classmethod
    def guessed_endian(klass, hdr):
        if hdr['an_integer'] < 256:
            return native_code
        return swapped_code

    @classmethod
    def default_structarr(klass, endianness=None):
        structarr = super().default_structarr(endianness)
        structarr['an_integer'] = 1
        structarr['a_str'] = b'a string'
        return structarr

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class"""
        return (klass._chk_integer, klass._chk_string)

    """ Check functions in format expected by BatteryRunner class """

    @staticmethod
    def _chk_integer(hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['an_integer'] == 1:
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'an_integer should be 1'
        if fix:
            hdr['an_integer'] = 1
            rep.fix_msg = 'set an_integer to 1'
        return hdr, rep

    @staticmethod
    def _chk_string(hdr, fix=False):
        rep = Report(HeaderDataError)
        hdr_str = str(hdr['a_str'])
        if hdr_str.lower() == hdr_str:
            return hdr, rep
        rep.problem_level = 20
        rep.problem_msg = 'a_str should be lower case'
        if fix:
            hdr['a_str'] = hdr_str.lower()
            rep.fix_msg = 'set a_str to lower case'
        return hdr, rep


class MyLabeledWrapStruct(LabeledWrapStruct, MyWrapStruct):
    _field_recoders = {}  # for recoding values for str


class TestMyWrapStruct(_TestWrapStructBase):
    """Test fake binary header defined at top of module"""

    header_class = MyWrapStruct

    def get_bad_bb(self):
        # A value for the binary block that should raise an error
        # Completely zeros binary block (nearly) always (fairly) bad
        return b'\x00' * self.header_class.template_dtype.itemsize

    def _set_something_into_hdr(self, hdr):
        # Called from test_bytes test method.  Specific to the header data type
        hdr['a_str'] = 'reggie'

    def test_empty(self):
        # Test contents of default header
        hdr = self.header_class()
        assert hdr['an_integer'] == 1
        assert hdr['a_str'] == b'a string'

    def test_str(self):
        hdr = self.header_class()
        s1 = str(hdr)
        assert len(s1) > 0
        assert 'an_integer' in s1
        assert 'a_str' in s1

    def test_copy(self):
        hdr = self.header_class()
        hdr2 = hdr.copy()
        assert hdr == hdr2
        self._set_something_into_hdr(hdr)
        assert hdr != hdr2
        self._set_something_into_hdr(hdr2)
        assert hdr == hdr2

    def test_checks(self):
        # Test header checks
        hdr_t = self.header_class()
        # _dxer just returns the diagnostics as a string
        # Default hdr is OK
        assert self._dxer(hdr_t) == ''
        # An integer should be 1
        hdr = hdr_t.copy()
        hdr['an_integer'] = 2
        assert self._dxer(hdr) == 'an_integer should be 1'
        # String should be lower case
        hdr = hdr_t.copy()
        hdr['a_str'] = 'My Name'
        assert self._dxer(hdr) == 'a_str should be lower case'

    def test_log_checks(self):
        # Test logging, fixing, errors for header checking
        # This is specific to the particular header type. Here we use the
        # pretent header defined at the top of this file
        HC = self.header_class
        hdr = HC()
        hdr['an_integer'] = 2  # severity 40
        fhdr, message, raiser = self.log_chk(hdr, 40)
        return
        assert fhdr['an_integer'] == 1
        assert message == 'an_integer should be 1; set an_integer to 1'
        pytest.raises(*raiser)
        # lower case string
        hdr = HC()
        hdr['a_str'] = 'Hello'  # severity = 20
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert message == 'a_str should be lower case; set a_str to lower case'
        pytest.raises(*raiser)

    def test_logger_error(self):
        # Check that we can reset the logger and error level
        # This is again specific to this pretend header
        HC = self.header_class
        hdr = HC()
        # Make a new logger
        str_io = StringIO()
        logger = logging.getLogger('test.logger')
        logger.setLevel(20)
        logger.addHandler(logging.StreamHandler(str_io))
        # Prepare something that needs fixing
        hdr['a_str'] = 'Fullness'  # severity 20
        log_cache = imageglobals.logger, imageglobals.error_level
        try:
            # Check log message appears in new logger
            imageglobals.logger = logger
            hdr.copy().check_fix()
            assert str_io.getvalue() == 'a_str should be lower case; set a_str to lower case\n'
            # Check that error_level in fact causes error to be raised
            imageglobals.error_level = 20
            with pytest.raises(HeaderDataError):
                hdr.copy().check_fix()
        finally:
            imageglobals.logger, imageglobals.error_level = log_cache


class TestMyLabeledWrapStruct(TestMyWrapStruct, _TestLabeledWrapStruct):
    header_class = MyLabeledWrapStruct

    def test_str(self):
        # Make sure not to overwrite class dictionary
        class MyHdr(self.header_class):
            _field_recoders = {}

        hdr = MyHdr()
        s1 = str(hdr)
        assert len(s1) > 0
        assert 'an_integer  : 1' in s1
        assert 'fullness of heart' not in s1
        rec = Recoder([[1, 'fullness of heart']], ('code', 'label'))
        hdr._field_recoders['an_integer'] = rec
        s2 = str(hdr)
        assert 'fullness of heart' in s2
        hdr['an_integer'] = 10
        s1 = str(hdr)
        assert '<unknown code 10>' in s1
