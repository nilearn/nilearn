# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for BatteryRunner and Report objects"""

import logging
from io import StringIO

import pytest

from ..batteryrunners import BatteryRunner, Report


# define some trivial functions as checks
def chk1(obj, fix=False):
    rep = Report(KeyError)
    if 'testkey' in obj:
        return obj, rep
    rep.problem_level = 20
    rep.problem_msg = 'no "testkey"'
    if fix:
        obj['testkey'] = 1
        rep.fix_msg = 'added "testkey"'
    return obj, rep


def chk2(obj, fix=False):
    # Can return different codes for different errors in same check
    rep = Report()
    try:
        ok = obj['testkey'] == 0
    except KeyError:
        rep.problem_level = 20
        rep.problem_msg = 'no "testkey"'
        rep.error = KeyError
        if fix:
            obj['testkey'] = 1
            rep.fix_msg = 'added "testkey"'
        return obj, rep
    if ok:
        return obj, rep
    rep.problem_level = 10
    rep.problem_msg = '"testkey" != 0'
    rep.error = ValueError
    if fix:
        rep.fix_msg = 'set "testkey" to 0'
        obj['testkey'] = 0
    return obj, rep


def chk_warn(obj, fix=False):
    rep = Report(KeyError)
    if not 'anotherkey' in obj:
        rep.problem_level = 30
        rep.problem_msg = 'no "anotherkey"'
        if fix:
            obj['anotherkey'] = 'a string'
            rep.fix_msg = 'added "anotherkey"'
    return obj, rep


def chk_error(obj, fix=False):
    rep = Report(KeyError)
    if not 'thirdkey' in obj:
        rep.problem_level = 40
        rep.problem_msg = 'no "thirdkey"'
        if fix:
            obj['anotherkey'] = 'a string'
            rep.fix_msg = 'added "anotherkey"'
    return obj, rep


def test_init_basic():
    # With no args, raise
    with pytest.raises(TypeError):
        BatteryRunner()
    # Len returns number of checks
    battrun = BatteryRunner((chk1,))
    assert len(battrun) == 1
    battrun = BatteryRunner((chk1, chk2))
    assert len(battrun) == 2


def test_init_report():
    rep = Report()
    assert rep == Report(Exception, 0, '', '')


def test_report_strings():
    rep = Report()
    assert rep.__str__() != ''
    assert rep.message == ''
    str_io = StringIO()
    rep.write_raise(str_io)
    assert str_io.getvalue() == ''
    rep = Report(ValueError, 20, 'msg', 'fix')
    rep.write_raise(str_io)
    assert str_io.getvalue() == ''
    rep.problem_level = 30
    rep.write_raise(str_io)
    assert str_io.getvalue() == 'Level 30: msg; fix\n'
    str_io.truncate(0)
    str_io.seek(0)
    # No fix string, no fix message
    rep.fix_msg = ''
    rep.write_raise(str_io)
    assert str_io.getvalue() == 'Level 30: msg\n'
    rep.fix_msg = 'fix'
    str_io.truncate(0)
    str_io.seek(0)
    # If we drop the level, nothing goes to the log
    rep.problem_level = 20
    rep.write_raise(str_io)
    assert str_io.getvalue() == ''
    # Unless we set the default log level in the call
    rep.write_raise(str_io, log_level=20)
    assert str_io.getvalue() == 'Level 20: msg; fix\n'
    str_io.truncate(0)
    str_io.seek(0)
    # If we set the error level down this low, we raise an error
    with pytest.raises(ValueError):
        rep.write_raise(str_io, 20)
    # But the log level wasn't low enough to do a log entry
    assert str_io.getvalue() == ''
    # Error still raised with lower log threshold, but now we do get a
    # log entry
    with pytest.raises(ValueError):
        rep.write_raise(str_io, 20, 20)
    assert str_io.getvalue() == 'Level 20: msg; fix\n'
    # If there's no error, we can't raise
    str_io.truncate(0)
    str_io.seek(0)
    rep.error = None
    rep.write_raise(str_io, 20)
    assert str_io.getvalue() == ''


def test_logging():
    rep = Report(ValueError, 20, 'msg', 'fix')
    str_io = StringIO()
    logger = logging.getLogger('test.logger')
    logger.setLevel(30)  # defaultish level
    logger.addHandler(logging.StreamHandler(str_io))
    rep.log_raise(logger)
    assert str_io.getvalue() == ''
    rep.problem_level = 30
    rep.log_raise(logger)
    assert str_io.getvalue() == 'msg; fix\n'
    str_io.truncate(0)
    str_io.seek(0)


def test_checks():
    battrun = BatteryRunner((chk1,))
    reports = battrun.check_only({})
    assert reports[0] == Report(KeyError, 20, 'no "testkey"', '')
    obj, reports = battrun.check_fix({})
    assert reports[0] == Report(KeyError, 20, 'no "testkey"', 'added "testkey"')
    assert obj == {'testkey': 1}
    battrun = BatteryRunner((chk1, chk2))
    reports = battrun.check_only({})
    assert reports[0] == Report(KeyError, 20, 'no "testkey"', '')
    assert reports[1] == Report(KeyError, 20, 'no "testkey"', '')
    obj, reports = battrun.check_fix({})
    # In the case of fix, the previous fix exposes a different error
    # Note, because obj is mutable, first and second point to modified
    # (and final) dictionary
    output_obj = {'testkey': 0}
    assert reports[0] == Report(KeyError, 20, 'no "testkey"', 'added "testkey"')
    assert reports[1] == Report(ValueError, 10, '"testkey" != 0', 'set "testkey" to 0')
    assert obj == output_obj
