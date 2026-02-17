"""Descriptor support for NIPY

Utilities to support special Python descriptors [1,2], in particular
:func:`~functools.cached_property`, which has been available in the Python
standard library since Python 3.8. We currently maintain aliases from
earlier names for this descriptor, specifically `OneTimeProperty` and `auto_attr`.

:func:`~functools.cached_property` creates properties that are computed once
and then stored as regular attributes. They can thus be evaluated
later in the object's life cycle, but once evaluated they become normal, static
attributes with no function call overhead on access or any other constraints.

A special ResetMixin class is provided to add a .reset() method to users who
may want to have their objects capable of resetting these computed properties
to their 'untriggered' state.

References
----------
[1] How-To Guide for Descriptors, Raymond
Hettinger. https://docs.python.org/howto/descriptor.html

[2] Python data model, https://docs.python.org/reference/datamodel.html
"""

from __future__ import annotations

from functools import cached_property

from nibabel.deprecated import deprecate_with_version

# -----------------------------------------------------------------------------
# Classes and Functions
# -----------------------------------------------------------------------------


class ResetMixin:
    """A Mixin class to add a .reset() method to users of cached_property.

    By default, cached properties, once computed, become static.  If they happen
    to depend on other parts of an object and those parts change, their values
    may now be invalid.

    This class offers a .reset() method that users can call *explicitly* when
    they know the state of their objects may have changed and they want to
    ensure that *all* their special attributes should be invalidated.  Once
    reset() is called, all their cached properties are reset to their
    :func:`~functools.cached_property` descriptors,
    and their accessor functions will be triggered again.

    .. warning::

       If a class has a set of attributes that are cached_property, but that
       can be initialized from any one of them, do NOT use this mixin!  For
       instance, UniformTimeSeries can be initialized with only sampling_rate
       and t0, sampling_interval and time are auto-computed.  But if you were
       to reset() a UniformTimeSeries, it would lose all 4, and there would be
       then no way to break the circular dependency chains.

       If this becomes a problem in practice (for our analyzer objects it
       isn't, as they don't have the above pattern), we can extend reset() to
       check for a _no_reset set of names in the instance which are meant to be
       kept protected.  But for now this is NOT done, so caveat emptor.

    Examples
    --------

    >>> class A(ResetMixin):
    ...     def __init__(self,x=1.0):
    ...         self.x = x
    ...
    ...     @cached_property
    ...     def y(self):
    ...         print('*** y computation executed ***')
    ...         return self.x / 2.0

    >>> a = A(10)

    About to access y twice, the second time no computation is done:

    >>> a.y
    *** y computation executed ***
    5.0
    >>> a.y
    5.0

    Changing x

    >>> a.x = 20

    a.y doesn't change to 10, since it is a static attribute:

    >>> a.y
    5.0

    We now reset a, and this will then force all auto attributes to recompute
    the next time we access them:

    >>> a.reset()

    About to access y twice again after reset():

    >>> a.y
    *** y computation executed ***
    10.0
    >>> a.y
    10.0
    """

    def reset(self) -> None:
        """Reset all cached_property attributes that may have fired already."""
        # To reset them, we simply remove them from the instance dict.  At that
        # point, it's as if they had never been computed.  On the next access,
        # the accessor function from the parent class will be called, simply
        # because that's how the python descriptor protocol works.
        for mname, mval in self.__class__.__dict__.items():
            if mname in self.__dict__ and isinstance(mval, cached_property):
                delattr(self, mname)


OneTimeProperty = cached_property
auto_attr = cached_property

# -----------------------------------------------------------------------------
# Deprecated API
# -----------------------------------------------------------------------------

# For backwards compatibility
setattr_on_read = deprecate_with_version(
    message='setattr_on_read has been renamed to auto_attr. Please use nibabel.onetime.auto_attr',
    since='3.2',
    until='5.0',
)(auto_attr)
