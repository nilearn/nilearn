"""Stream-like reader for packed data"""

from struct import Struct

_ENDIAN_CODES = '@=<>!'


class Unpacker:
    """Class to unpack values from buffer object

    The buffer object is usually a string. Caches compiled :mod:`struct`
    format strings so that repeated unpacking with the same format
    string should be faster than using ``struct.unpack`` directly.

    Examples
    --------
    >>> a = b'1234567890'
    >>> upk = Unpacker(a)
    >>> upk.unpack('2s') == (b'12',)
    True
    >>> upk.unpack('2s') == (b'34',)
    True
    >>> upk.ptr
    4
    >>> upk.read(3) == b'567'
    True
    >>> upk.ptr
    7
    """

    def __init__(self, buf, ptr=0, endian=None):
        """Initialize unpacker

        Parameters
        ----------
        buf : buffer
           object implementing buffer protocol (e.g. str)
        ptr : int, optional
           offset at which to begin reads from `buf`
        endian : None or str, optional
           endian code to prepend to format, as for ``unpack`` endian
           codes.  None (the default) corresponds to the default
           behavior of ``struct`` - assuming system endian unless you
           specify the byte order specifically in the format string
           passed to ``unpack``
        """
        self.buf = buf
        self.ptr = ptr
        self.endian = endian
        self._cache = {}

    def unpack(self, fmt):
        """Unpack values from contained buffer

        Unpacks values from ``self.buf`` and updates ``self.ptr`` to the
        position after the read data.

        Parameters
        ----------
        fmt : str
           format string as for ``unpack``

        Returns
        -------
        values : tuple
           values as unpacked from ``self.buf`` according to `fmt`
        """
        # try and get a struct corresponding to the format string from
        # the cache
        pkst = self._cache.get(fmt)
        if pkst is None:  # struct not in cache
            # if we've not got a default endian, or the format has an
            # explicit endianness, then we make a new struct directly
            # from the format string
            if self.endian is None or fmt[0] in _ENDIAN_CODES:
                pkst = Struct(fmt)
            else:  # we're going to modify the endianness with our
                # default.
                endian_fmt = self.endian + fmt
                pkst = Struct(endian_fmt)
                # add an entry in the cache for the modified format
                # string as well as (below) the unmodified format
                # string, in case we get a format string with the same
                # endianness as default, but specified explicitly.
                self._cache[endian_fmt] = pkst
            self._cache[fmt] = pkst
        values = pkst.unpack_from(self.buf, self.ptr)
        self.ptr += pkst.size
        return values

    def read(self, n_bytes=-1):
        """Return byte string of length `n_bytes` at current position

        Returns sub-string from ``self.buf`` and updates ``self.ptr`` to the
        position after the read data.

        Parameters
        ----------
        n_bytes : int, optional
           number of bytes to read.  Can be -1 (the default) in which
           case we return all the remaining bytes in ``self.buf``

        Returns
        -------
        s : byte string
        """
        start = self.ptr
        if n_bytes == -1:
            end = len(self.buf)
        else:
            end = start + n_bytes
        self.ptr = end
        return self.buf[start:end]
