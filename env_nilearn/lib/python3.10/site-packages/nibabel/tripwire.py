"""Class to raise error for missing modules or other misfortunes"""

from typing import Any


class TripWireError(AttributeError):
    """Exception if trying to use TripWire object"""

    # Has to be subclass of AttributeError, to work round Python 3.5 inspection
    # for doctests.  Python 3.5 looks for a ``__wrapped__`` attribute during
    # initialization of doctests, and only allows AttributeError as signal this
    # is not present.


def is_tripwire(obj: Any) -> bool:
    """Returns True if `obj` appears to be a TripWire object

    Examples
    --------
    >>> is_tripwire(object())
    False
    >>> is_tripwire(TripWire('some message'))
    True
    """
    try:
        obj.any_attribute
    except TripWireError:
        return True
    except Exception:
        pass
    return False


class TripWire:
    """Class raising error if used

    Standard use is to proxy modules that we could not import

    Examples
    --------
    >>> a_module = TripWire('We do not have a_module')
    >>> a_module.do_silly_thing('with silly string') #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TripWireError: We do not have a_module
    """

    def __init__(self, msg: str) -> None:
        self._msg = msg

    def __getattr__(self, attr_name: str) -> Any:
        """Raise informative error accessing attributes"""
        raise TripWireError(self._msg)
