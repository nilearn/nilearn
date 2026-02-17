"""Array writer objects

Array writers have init signature::

    def __init__(self, array, out_dtype=None)

and methods

* scaling_needed() - returns True if array requires scaling for write
* finite_range() - returns min, max of self.array
* to_fileobj(fileobj, offset=None, order='F')

They must have attributes / properties of:

* array
* out_dtype
* has_nan

They may have attributes:

* slope
* inter

They are designed to write arrays to a fileobj with reasonable memory
efficiency.

Array writers may be able to scale the array or apply an intercept, or do
something else to make sense of conversions between float and int, or between
larger ints and smaller.
"""

import numpy as np

from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range


class WriterError(Exception):
    pass


class ScalingError(WriterError):
    pass


class ArrayWriter:
    def __init__(self, array, out_dtype=None, **kwargs):
        r"""Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        \*\*kwargs : keyword arguments
            This class processes only:

            * nan2zero : bool, optional
              Whether to set NaN values to 0 when writing integer output.
              Defaults to True.  If False, NaNs get converted with numpy
              ``astype``, and the behavior is undefined.  Ignored for floating
              point output.
            * check_scaling : bool, optional
              If True, check if scaling needed and raise error if so. Default
              is True

        Examples
        --------
        >>> arr = np.array([0, 255], np.uint8)
        >>> aw = ArrayWriter(arr)
        >>> aw = ArrayWriter(arr, np.int8) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        WriterError: Scaling needed but cannot scale
        >>> aw = ArrayWriter(arr, np.int8, check_scaling=False)
        """
        nan2zero = kwargs.pop('nan2zero', True)
        check_scaling = kwargs.pop('check_scaling', True)
        self._array = np.asanyarray(array)
        arr_dtype = self._array.dtype
        if out_dtype is None:
            out_dtype = arr_dtype
        else:
            out_dtype = np.dtype(out_dtype)
        self._out_dtype = out_dtype
        self._finite_range = None
        self._has_nan = None
        self._nan2zero = nan2zero
        if check_scaling and self.scaling_needed():
            raise WriterError('Scaling needed but cannot scale')

    def scaling_needed(self):
        """Checks if scaling is needed for input array

        Raises WriterError if no scaling possible.

        The rules are in the code, but:

        * If numpy will cast, return False (no scaling needed)
        * If input or output is an object or structured type, raise
        * If input is complex, raise
        * If the output is float, return False
        * If the input array is all zero, return False
        * By now we are casting to (u)int. If the input type is a float, return
          True (we do need scaling)
        * Now input and output types are (u)ints. If the min and max in the
          data are within range of the output type, return False
        * Otherwise return True
        """
        data = self._array
        arr_dtype = data.dtype
        out_dtype = self._out_dtype
        # There's a bug in np.can_cast (at least up to and including 1.6.1)
        # such that any structured output type passes.  Check for this first.
        if 'V' in (arr_dtype.kind, out_dtype.kind):
            if arr_dtype == out_dtype:
                return False
            raise WriterError('Cannot cast to or from non-numeric types')
        if np.can_cast(arr_dtype, out_dtype):
            return False
        # Direct casting for complex output from any numeric type
        if out_dtype.kind == 'c':
            return False
        if arr_dtype.kind == 'c':
            raise WriterError('Cannot cast complex types to non-complex')
        # Direct casting for float output from any non-complex numeric type
        if out_dtype.kind == 'f':
            return False
        # Now we need to look at the data for special cases
        if data.size == 0:
            return False
        mn, mx = self.finite_range()  # this is cached
        if (mn, mx) == (0, 0):
            # Data all zero
            return False
        # Floats -> (u)ints always need scaling
        if arr_dtype.kind == 'f':
            return True
        # (u)int input, (u)int output
        assert arr_dtype.kind in 'iu' and out_dtype.kind in 'iu'
        info = np.iinfo(out_dtype)
        # No scaling needed if data already fits in output type
        # But note - we need to convert to ints, to avoid conversion to float
        # during comparisons, and therefore int -> float conversions which are
        # not exact.  Only a problem for uint64 though.
        if int(mn) >= int(info.min) and int(mx) <= int(info.max):
            return False
        return True

    @property
    def array(self):
        """Return array from arraywriter"""
        return self._array

    @property
    def out_dtype(self):
        """Return `out_dtype` from arraywriter"""
        return self._out_dtype

    @property
    def has_nan(self):
        """True if array has NaNs"""
        # Structured types raise an error for finite range; don't run finite
        # range unless we have to.
        if self._has_nan is None:
            if self._array.dtype.kind in 'fc':
                self.finite_range()
            else:
                self._has_nan = False
        return self._has_nan

    def finite_range(self):
        """Return (maybe cached) finite range of data array"""
        if self._finite_range is None:
            mn, mx, has_nan = finite_range(self._array, True)
            self._finite_range = (mn, mx)
            self._has_nan = has_nan
        return self._finite_range

    def _needs_nan2zero(self):
        """True if nan2zero check needed for writing array"""
        return (
            self._nan2zero
            and self._array.dtype.kind in 'fc'
            and self.out_dtype.kind in 'iu'
            and self.has_nan
        )

    def to_fileobj(self, fileobj, order='F'):
        """Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
        array_to_file(
            self._array,
            fileobj,
            self._out_dtype,
            offset=None,
            mn=None,
            mx=None,
            order=order,
            nan2zero=self._needs_nan2zero(),
        )


class SlopeArrayWriter(ArrayWriter):
    """ArrayWriter that can use scalefactor for writing arrays

    The scalefactor allows the array writer to write floats to int output
    types, and rescale larger ints to smaller.  It can therefore lose
    precision.

    It extends the ArrayWriter class with attribute:

    * slope

    and methods:

    * reset() - reset slope to default (not adapted to self.array)
    * calc_scale() - calculate slope to best write self.array
    """

    def __init__(self, array, out_dtype=None, calc_scale=True, scaler_dtype=np.float32, **kwargs):
        r"""Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        calc_scale : {True, False}, optional
            Whether to calculate scaling for writing `array` on initialization.
            If False, then you can calculate this scaling with
            ``obj.calc_scale()`` - see examples
        scaler_dtype : dtype-like, optional
            specifier for numpy dtype for scaling
        \*\*kwargs : keyword arguments
            This class processes only:

            * nan2zero : bool, optional
              Whether to set NaN values to 0 when writing integer output.
              Defaults to True.  If False, NaNs get converted with numpy
              ``astype``, and the behavior is undefined.  Ignored for floating
              point output.

        Examples
        --------
        >>> arr = np.array([0, 254], np.uint8)
        >>> aw = SlopeArrayWriter(arr)
        >>> aw.slope
        1.0
        >>> aw = SlopeArrayWriter(arr, np.int8)
        >>> aw.slope
        2.0
        >>> aw = SlopeArrayWriter(arr, np.int8, calc_scale=False)
        >>> aw.slope
        1.0
        >>> aw.calc_scale()
        >>> aw.slope
        2.0
        """
        nan2zero = kwargs.pop('nan2zero', True)
        self._array = np.asanyarray(array)
        arr_dtype = self._array.dtype
        if out_dtype is None:
            out_dtype = arr_dtype
        else:
            out_dtype = np.dtype(out_dtype)
        self._out_dtype = out_dtype
        self.scaler_dtype = np.dtype(scaler_dtype)
        self.reset()
        self._nan2zero = nan2zero
        self._has_nan = None
        if calc_scale:
            self.calc_scale()

    def scaling_needed(self):
        """Checks if scaling is needed for input array

        Raises WriterError if no scaling possible.

        The rules are in the code, but:

        * If numpy will cast, return False (no scaling needed)
        * If input or output is an object or structured type, raise
        * If input is complex, raise
        * If the output is float, return False
        * If the input array is all zero, return False
        * If there is no finite value, return False (the writer will strip the
          non-finite values)
        * By now we are casting to (u)int. If the input type is a float, return
          True (we do need scaling)
        * Now input and output types are (u)ints. If the min and max in the
          data are within range of the output type, return False
        * Otherwise return True
        """
        if not super().scaling_needed():
            return False
        mn, mx = self.finite_range()  # this is cached
        # No finite data - no scaling needed
        return (mn, mx) != (np.inf, -np.inf)

    def reset(self):
        """Set object to values before any scaling calculation"""
        self.slope = 1.0
        self._finite_range = None
        self._scale_calced = False

    def _get_slope(self):
        return self._slope

    def _set_slope(self, val):
        self._slope = np.squeeze(self.scaler_dtype.type(val))

    slope = property(_get_slope, _set_slope, None, 'get/set slope')

    def calc_scale(self, force=False):
        """Calculate / set scaling for floats/(u)ints to (u)ints"""
        # If we've run already, return unless told otherwise
        if not force and self._scale_calced:
            return
        self.reset()
        if not self.scaling_needed():
            return
        self._do_scaling()
        self._scale_calced = True

    def _writing_range(self):
        """Finite range for thresholding on write"""
        if self._out_dtype.kind in 'iu' and self._array.dtype.kind == 'f':
            mn, mx = self.finite_range()
            if (mn, mx) == (np.inf, -np.inf):  # no finite data
                mn, mx = 0, 0
            return mn, mx
        return None, None

    def to_fileobj(self, fileobj, order='F'):
        """Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
        mn, mx = self._writing_range()
        array_to_file(
            self._array,
            fileobj,
            self._out_dtype,
            offset=None,
            divslope=self.slope,
            mn=mn,
            mx=mx,
            order=order,
            nan2zero=self._needs_nan2zero(),
        )

    def _do_scaling(self):
        arr = self._array
        out_dtype = self._out_dtype
        assert out_dtype.kind in 'iu'
        mn, mx = self.finite_range()
        if arr.dtype.kind == 'f':
            # Float to (u)int scaling
            # Need to take nan2zero value into account for scaling
            if self._nan2zero and self.has_nan:
                mn = min(mn, 0)
                mx = max(mx, 0)
            self._range_scale(mn, mx)
            return
        # (u)int to (u)int
        info = np.iinfo(out_dtype)
        out_max, out_min = info.max, info.min
        # If left as int64, uint64, comparisons will default to floats, and
        # these are inexact for > 2**53 - so convert to int
        if int(mx) <= int(out_max) and int(mn) >= int(out_min):
            # already in range
            return
        # (u)int to (u)int scaling
        self._iu2iu()

    def _iu2iu(self):
        # (u)int to (u)int scaling
        mn, mx = self.finite_range()
        out_dt = self._out_dtype
        if out_dt.kind == 'u':
            # We're checking for a sign flip.  This can only work for uint
            # output, because, for int output, the abs min of the type is
            # greater than the abs max, so the data either fits into the range
            # (tested for in _do_scaling), or this test can't pass. Need abs
            # that deals with max neg ints. abs problem only arises when all
            # the data is set to max neg integer value
            o_min, o_max = shared_range(self.scaler_dtype, out_dt)
            if mx <= 0 and int_abs(mn) <= int(o_max):  # sign flip enough?
                # -1.0 * arr will be in scaler_dtype precision
                self.slope = -1.0
                return
        self._range_scale(mn, mx)

    def _range_scale(self, in_min, in_max):
        """Calculate scaling based on data range and output type"""
        out_dtype = self._out_dtype
        info = type_info(out_dtype)
        out_min, out_max = info['min'], info['max']
        big_float = best_float()
        if out_dtype.kind == 'f':
            # But we want maximum precision for the calculations. Casting will
            # not lose precision because min/max are of fp type.
            out_min, out_max = np.array((out_min, out_max), dtype=big_float)
        else:  # (u)int
            out_min, out_max = (big_float(v) for v in (out_min, out_max))
        if self._out_dtype.kind == 'u':
            if in_min < 0 and in_max > 0:
                raise WriterError(
                    'Cannot scale negative and positive numbers to uint without intercept'
                )
            if in_max <= 0:  # All input numbers <= 0
                self.slope = in_min / out_max
            else:  # All input numbers > 0
                self.slope = in_max / out_max
            return
        # Scaling to int. We need the bigger slope of (in_min/out_min) and
        # (in_max/out_max). If in_min or in_max is the wrong side of 0, that
        # will make these negative and so they won't worry us
        mx_slope = in_max / out_max
        mn_slope = in_min / out_min
        self.slope = np.max([mx_slope, mn_slope])


class SlopeInterArrayWriter(SlopeArrayWriter):
    """Array writer that can use slope and intercept to scale array

    The writer can subtract an intercept, and divided by a slope, in order to
    be able to convert floating point values into a (u)int range, or to convert
    larger (u)ints to smaller.

    It extends the ArrayWriter class with attributes:

    * inter
    * slope

    and methods:

    * reset() - reset inter, slope to default (not adapted to self.array)
    * calc_scale() - calculate inter, slope to best write self.array
    """

    def __init__(self, array, out_dtype=None, calc_scale=True, scaler_dtype=np.float32, **kwargs):
        r"""Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        calc_scale : {True, False}, optional
            Whether to calculate scaling for writing `array` on initialization.
            If False, then you can calculate this scaling with
            ``obj.calc_scale()`` - see examples
        scaler_dtype : dtype-like, optional
            specifier for numpy dtype for slope, intercept
        \*\*kwargs : keyword arguments
            This class processes only:

            * nan2zero : bool, optional
              Whether to set NaN values to 0 when writing integer output.
              Defaults to True.  If False, NaNs get converted with numpy
              ``astype``, and the behavior is undefined.  Ignored for floating
              point output.

        Examples
        --------
        >>> arr = np.array([0, 255], np.uint8)
        >>> aw = SlopeInterArrayWriter(arr)
        >>> aw.slope, aw.inter
        (1.0, 0.0)
        >>> aw = SlopeInterArrayWriter(arr, np.int8)
        >>> (aw.slope, aw.inter) == (1.0, 128)
        True
        >>> aw = SlopeInterArrayWriter(arr, np.int8, calc_scale=False)
        >>> aw.slope, aw.inter
        (1.0, 0.0)
        >>> aw.calc_scale()
        >>> (aw.slope, aw.inter) == (1.0, 128)
        True
        """
        super().__init__(array, out_dtype, calc_scale, scaler_dtype, **kwargs)

    def reset(self):
        """Set object to values before any scaling calculation"""
        super().reset()
        self.inter = 0.0

    def _get_inter(self):
        return self._inter

    def _set_inter(self, val):
        self._inter = np.squeeze(self.scaler_dtype.type(val))

    inter = property(_get_inter, _set_inter, None, 'get/set inter')

    def to_fileobj(self, fileobj, order='F'):
        """Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
        mn, mx = self._writing_range()
        array_to_file(
            self._array,
            fileobj,
            self._out_dtype,
            offset=None,
            intercept=self.inter,
            divslope=self.slope,
            mn=mn,
            mx=mx,
            order=order,
            nan2zero=self._needs_nan2zero(),
        )

    def _iu2iu(self):
        # (u)int to (u)int
        mn, mx = (int(v) for v in self.finite_range())
        # range may be greater than the largest integer for this type.
        out_dtype = self._out_dtype
        # Options in this method are scaling using intercept only.  These will
        # have to pass through ``self.scaler_dtype`` (because the intercept is
        # in this type).
        o_min, o_max = (int(v) for v in shared_range(self.scaler_dtype, out_dtype))
        type_range = o_max - o_min
        mn2mx = mx - mn
        if mn2mx <= type_range:  # might offset be enough?
            if o_min == 0:  # uint output - take min to 0
                # decrease offset with floor_exact, meaning mn >= t_min after
                # subtraction.  But we may have pushed the data over t_max,
                # which we check below
                inter = floor_exact(mn - o_min, self.scaler_dtype)
            else:  # int output - take midpoint to 0
                # ceil below increases inter, pushing scale up to 0.5 towards
                # -inf, because ints have abs min == abs max + 1
                midpoint = mn + int(np.ceil(mn2mx / 2.0))
                # Floor exact decreases inter, so pulling scaled values more
                # positive. This may make mx - inter > t_max
                inter = floor_exact(midpoint, self.scaler_dtype)
            # Need to check still in range after floor_exact-ing
            int_inter = int(inter)
            assert mn - int_inter >= o_min
            if mx - int_inter <= o_max:
                self.inter = inter
                return
        # Try slope options (sign flip) and then range scaling
        super()._iu2iu()

    def _range_scale(self, in_min, in_max):
        """Calculate scaling, intercept based on data range and output type"""
        if in_max == in_min:  # Only one number in array
            self.slope = 1.0
            self.inter = in_min
            return
        big_float = best_float()
        in_dtype = self._array.dtype
        out_dtype = self._out_dtype
        working_dtype = self.scaler_dtype
        if in_dtype.kind == 'f':  # Already floats
            # float64 and below cast correctly to longdouble.  Longdouble needs
            # no casting
            in_min, in_max = np.array([in_min, in_max], dtype=big_float)
            in_range = np.diff([in_min, in_max])
        else:  # max possible (u)int range is 2**64-1 (int64, uint64)
            # On windows longdouble is the same as double so in_range will be 2**64 -
            # thus overestimating slope slightly.  Casting to int needed to allow
            # in_max-in_min to be larger than the largest (u)int value
            in_min, in_max = int(in_min), int(in_max)
            in_range = big_float(in_max - in_min)
            # Cast to float for later processing.
            in_min, in_max = (big_float(v) for v in (in_min, in_max))
        if out_dtype.kind == 'f':
            # Type range, these are also floats
            info = type_info(out_dtype)
            out_min, out_max = info['min'], info['max']
        else:
            # Use shared range to avoid rounding to values outside range. This
            # doesn't matter much except for the case of nan2zero were we need
            # to be able to represent the scaled zero correctly in order not to
            # raise an error when writing
            out_min, out_max = shared_range(working_dtype, out_dtype)
            out_min, out_max = np.array((out_min, out_max), dtype=big_float)
        # We want maximum precision for the calculations. Casting will not lose
        # precision because min/max are of fp type.
        assert [v.dtype.kind for v in (out_min, out_max)] == ['f', 'f']
        out_range = out_max - out_min
        """
        Think of the input values as a line starting (left) at in_min and
        ending (right) at in_max.

        The output values will be a line starting at out_min and ending at
        out_max.

        We are going to match the input line to the output line by subtracting
        `inter` then dividing by `slope`.

        Slope must scale the input line to have the same length as the output
        line.  We find this scale factor by dividing the input range (line
        length) by the output range (line length)
        """
        slope = in_range / out_range
        """
        Now we know the slope, we need the intercept.  The intercept will be
        such that:

            (in_min - inter) / slope = out_min

        Solving for the intercept:

            inter = in_min - out_min * slope

        We can also flip the sign of the slope.  In that case we match the
        in_max to the out_min:

            (in_max - inter_flipped) / -slope = out_min
            inter_flipped = in_max + out_min * slope

        When we reconstruct the data, we're going to do:

            data = saved_data * slope + inter

        We can't change the range of the saved data (the whole range of the
        integer type) or the range of the output data (the values we input). We
        can change the intermediate values ``saved_data * slope`` by choosing
        the sign of the slope to match the in_min or in_max to the left or
        right end of the saved data range.

        If the out_dtype is signed int, then abs(out_min) = abs(out_max) + 1
        and the absolute value and therefore precision for values at the left
        and right of the saved data range are very similar (e.g. -128 * slope,
        127 * slope respectively).

        If the out_dtype is unsigned int, then the absolute value at the left
        is 0 and the precision is much higher than for the right end of the
        range (e.g. 0 * slope, 255 * slope).

        If the out_dtype is unsigned int then we choose the sign of the slope
        to match the smaller of the in_min, in_max to the zero end of the saved
        range.
        """
        if out_min == 0 and np.abs(in_max) < np.abs(in_min):
            inter = in_max + out_min * slope
            slope *= -1
        else:
            inter = in_min - out_min * slope
        # slope, inter properties force scaling_dtype cast
        self.inter = inter
        self.slope = slope
        if not np.all(np.isfinite([self.slope, self.inter])):
            raise ScalingError('Slope / inter not both finite')
        # Check nan fill value
        if not (0 in (in_min, in_max) and self._nan2zero and self.has_nan):
            return
        nan_fill_f = -self.inter / self.slope
        nan_fill_i = np.rint(nan_fill_f)
        if nan_fill_i == np.array(nan_fill_i, dtype=out_dtype):
            return
        # recalculate intercept using dtype of inter, scale
        self.inter = -np.clip(nan_fill_f, out_min, out_max) * self.slope
        nan_fill_i = np.rint(-self.inter / self.slope)
        assert nan_fill_i == np.array(nan_fill_i, dtype=out_dtype)


def get_slope_inter(writer):
    """Return slope, intercept from array writer object

    Parameters
    ----------
    writer : ArrayWriter instance

    Returns
    -------
    slope : scalar
        slope in `writer` or 1.0 if not present
    inter : scalar
        intercept in `writer` or 0.0 if not present

    Examples
    --------
    >>> arr = np.arange(10)
    >>> get_slope_inter(ArrayWriter(arr))
    (1.0, 0.0)
    >>> get_slope_inter(SlopeArrayWriter(arr))
    (1.0, 0.0)
    >>> get_slope_inter(SlopeInterArrayWriter(arr))
    (1.0, 0.0)
    """
    try:
        slope = writer.slope
    except AttributeError:
        slope = 1.0
    try:
        inter = writer.inter
    except AttributeError:
        inter = 0.0
    return slope, inter


def make_array_writer(data, out_type, has_slope=True, has_intercept=True, **kwargs):
    r"""Make array writer instance for array `data` and output type `out_type`

    Parameters
    ----------
    data : array-like
        array for which to create array writer
    out_type : dtype-like
        input to numpy dtype to specify array writer output type
    has_slope : {True, False}
        If True, array write can use scaling to adapt the array to `out_type`
    has_intercept : {True, False}
        If True, array write can use intercept to adapt the array to `out_type`
    \*\*kwargs : other keyword arguments
        to pass to the arraywriter class

    Returns
    -------
    writer : arraywriter instance
        Instance of array writer, with class adapted to `has_intercept` and
        `has_slope`.

    Examples
    --------
    >>> aw = make_array_writer(np.arange(10), np.uint8, True, True)
    >>> type(aw) == SlopeInterArrayWriter
    True
    >>> aw = make_array_writer(np.arange(10), np.uint8, True, False)
    >>> type(aw) == SlopeArrayWriter
    True
    >>> aw = make_array_writer(np.arange(10), np.uint8, False, False)
    >>> type(aw) == ArrayWriter
    True
    """
    data = np.asarray(data)
    if has_intercept and not has_slope:
        raise ValueError('Cannot handle intercept without slope')
    if has_intercept:
        return SlopeInterArrayWriter(data, out_type, **kwargs)
    if has_slope:
        return SlopeArrayWriter(data, out_type, **kwargs)
    return ArrayWriter(data, out_type, **kwargs)
