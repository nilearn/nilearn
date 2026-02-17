# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Battery runner classes and Report classes

These classes / objects are for generic checking / fixing batteries

The ``BatteryRunner`` class will run a series of checks on a single
object.

A check is a callable, of signature ``func(obj, fix=False)`` which
returns a tuple ``(obj, Report)`` for ``func(obj, False)`` or
``func(obj, True)``, where the obj may be a modified object, or a
different object, if ``fix==True``.

To run checks only, and return problem report objects:

>>> from nibabel.batteryrunners import BatteryRunner, Report
>>> def chk(obj, fix=False): # minimal check
...     return obj, Report()
>>> btrun = BatteryRunner((chk,))
>>> reports = btrun.check_only('a string')

To run checks and fixes, returning fixed object and problem report
sequence, with possible fix messages:

>>> fixed_obj, report_seq = btrun.check_fix('a string')

Reports are iterable things, where the elements in the iterations are
``Problems``, with attributes ``error``, ``problem_level``,
``problem_msg``, and possibly empty ``fix_msg``.  The ``problem_level``
is an integer, giving the level of problem, from 0 (no problem) to 50
(very bad problem).  The levels follow the log levels from the logging
module (e.g 40 equivalent to "error" level, 50 to "critical").  The
``error`` can be one of ``None`` if no error to suggest, or an Exception
class that the user might consider raising for this situation.  The
``problem_msg`` and ``fix_msg`` are human readable strings that should
explain what happened.

=======================
 More about ``checks``
=======================

Checks are callables returning objects and reports, like ``chk`` below,
such that::

   obj, report = chk(obj, fix=False)
   obj, report = chk(obj, fix=True)

For example, for the Analyze header, we need to check the datatype::

    def chk_datatype(hdr, fix=True):
        rep = Report(hdr, HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dtype = AnalyzeHeader._data_type_codes.dtype[code]
        except KeyError:
            rep.problem_level = 40
            rep.problem_msg = 'data code not recognized'
        else:
            if dtype.type is np.void:
                rep.problem_level = 40
                rep.problem_msg = 'data code not supported'
            else:
                return hdr, rep
        if fix:
            rep.fix_problem_msg = 'not attempting fix'
        return hdr, rep

or the bitpix::

    def chk_bitpix(hdr, fix=True):
        rep = Report(HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dt = AnalyzeHeader._data_type_codes.dtype[code]
        except KeyError:
            rep.problem_level = 10
            rep.problem_msg = 'no valid datatype to fix bitpix'
            return hdr, rep
        bitpix = dt.itemsize * 8
        if bitpix == hdr['bitpix']:
            return hdr, rep
        rep.problem_level = 10
        rep.problem_msg = 'bitpix does not match datatype')
        if fix:
            hdr['bitpix'] = bitpix # inplace modification
            rep.fix_msg = 'setting bitpix to match datatype'
        return hdr, ret

or the pixdims::

    def chk_pixdims(hdr, fix=True):
        rep = Report(hdr, HeaderDataError)
        if not np.any(hdr['pixdim'][1:4] < 0):
            return hdr, rep
        rep.problem_level = 40
        rep.problem_msg = 'pixdim[1,2,3] should be positive'
        if fix:
            hdr['pixdim'][1:4] = np.abs(hdr['pixdim'][1:4])
            rep.fix_msg = 'setting to abs of pixdim values'
        return hdr, rep
"""


class BatteryRunner:
    """Class to run set of checks"""

    def __init__(self, checks):
        """Initialize instance from sequence of `checks`

        Parameters
        ----------
        checks : sequence
           sequence of checks, where checks are callables matching
           signature ``obj, rep = chk(obj, fix=False)``.  Checks are run
           in the order they are passed.

        Examples
        --------
        >>> def chk(obj, fix=False): # minimal check
        ...     return obj, Report()
        >>> btrun = BatteryRunner((chk,))
        """
        self._checks = checks

    def check_only(self, obj):
        """Run checks on `obj` returning reports

        Parameters
        ----------
        obj : anything
           object on which to run checks

        Returns
        -------
        reports : sequence
           sequence of report objects reporting on result of running
           checks (without fixes) on `obj`
        """
        reports = []
        for check in self._checks:
            obj, rep = check(obj, False)
            reports.append(rep)
        return reports

    def check_fix(self, obj):
        """Run checks, with fixes, on `obj` returning `obj`, reports

        Parameters
        ----------
        obj : anything
           object on which to run checks, fixes

        Returns
        -------
        obj : anything
           possibly modified or replaced `obj`, after fixes
        reports : sequence
           sequence of reports on checks, fixes
        """
        reports = []
        for check in self._checks:
            obj, report = check(obj, True)
            reports.append(report)
        return obj, reports

    def __len__(self):
        return len(self._checks)


class Report:
    def __init__(self, error=Exception, problem_level=0, problem_msg='', fix_msg=''):
        """Initialize report with values

        Parameters
        ----------
        error : None or Exception
           Error to raise if raising error for this check.  If None,
           no error can be raised for this check (it was probably
           normal).
        problem_level : int
           level of problem.  From 0 (no problem) to 50 (severe
           problem).  If the report originates from a fix, then this
           is the level of the problem remaining after the fix.
           Default is 0
        problem_msg : string
           String describing problem detected. Default is ''
        fix_msg : string
           String describing any fix applied.  Default is ''.

        Examples
        --------
        >>> rep = Report()
        >>> rep.problem_level
        0
        >>> rep = Report(TypeError, 10)
        >>> rep.problem_level
        10
        """
        self.error = error
        self.problem_level = problem_level
        self.problem_msg = problem_msg
        self.fix_msg = fix_msg

    def __getstate__(self):
        """State that defines object

        Returns
        -------
        tup : tuple
        """
        return self.error, self.problem_level, self.problem_msg, self.fix_msg

    def __eq__(self, other):
        """are two BatteryRunner-like objects equal?

        Parameters
        ----------
        other : object
           report-like object to test equality

        Examples
        --------
        >>> rep = Report(problem_level=10)
        >>> rep2 = Report(problem_level=10)
        >>> rep == rep2
        True
        >>> rep3 = Report(problem_level=20)
        >>> rep == rep3
        False
        """
        return self.__getstate__() == other.__getstate__()

    def __ne__(self, other):
        """are two BatteryRunner-like objects not equal?

        See docstring for __eq__
        """
        return not self == other

    def __str__(self):
        """Printable string for object"""
        return self.__dict__.__str__()

    @property
    def message(self):
        """formatted message string, including fix message if present"""
        if self.fix_msg:
            return f'{self.problem_msg}; {self.fix_msg}'
        return self.problem_msg

    def log_raise(self, logger, error_level=40):
        """Log problem, raise error if problem >= `error_level`

        Parameters
        ----------
        logger : log
           log object, implementing ``log`` method
        error_level : int, optional
           If ``self.problem_level`` >= `error_level`, raise error
        """
        logger.log(self.problem_level, self.message)
        if self.problem_level and self.problem_level >= error_level:
            if self.error:
                raise self.error(self.problem_msg)

    def write_raise(self, stream, error_level=40, log_level=30):
        """Write report to `stream`

        Parameters
        ----------
        stream : file-like
           implementing ``write`` method
        error_level : int, optional
           level at which to raise error for problem detected in
           ``self``
        log_level : int, optional
           Such that if `log_level` is >= ``self.problem_level`` we
           write the report to `stream`, otherwise we write nothing.
        """
        if self.problem_level >= log_level:
            stream.write(f'Level {self.problem_level}: {self.message}\n')
        if self.problem_level and self.problem_level >= error_level:
            if self.error:
                raise self.error(self.problem_msg)
