"""
Mixin for cache with joblib
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import warnings

from sklearn.externals.joblib import Memory

memory_classes = (Memory, )

try:
    from joblib import Memory as JoblibMemory
    memory_classes = (Memory, JoblibMemory)
except ImportError:
    pass


def cache(func, memory, ref_memory_level=2, memory_level=1, **kwargs):
    """ Return a joblib.Memory object.

    The memory_level determines the level above which the wrapped
    function output is cached. By specifying a numeric value for
    this level, the user can to control the amount of cache memory
    used. This function will cache the function call or not
    depending on the cache level.

    Parameters
    ----------
    func: function
        The function which output is to be cached.

    memory: instance of joblib.Memory or string
        Used to cache the function call.

    ref_memory_level: int
        The reference memory_level used to determine if function call must
        be cached or not (if memory_level is larger than ref_memory_level
        the function is cached)

    memory_level: int
        The memory_level from which caching must be enabled for the wrapped
        function.

    kwargs: keyword arguments
        The keyword arguments passed to memory.cache

    Returns
    -------
    mem: joblib.MemorizedFunc
        object that wraps the function func. This object may be
        a no-op, if the requested level is lower than the value given
        to _cache()). For consistency, a joblib.Memory object is always
        returned.
    """

    if ref_memory_level <= memory_level or memory is None:
        memory = Memory(cachedir=None)
    else:
        memory = memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)
        if not isinstance(memory, memory_classes):
            raise TypeError("'memory' argument must be a string or a "
                            "joblib.Memory object. "
                            "%s %s was given." % (memory, type(memory)))
        if memory.cachedir is None:
            warnings.warn("Caching has been enabled (memory_level = %d) "
                          "but no Memory object or path has been provided"
                          " (parameter memory). Caching deactivated for "
                          "function %s." %
                          (ref_memory_level, func.func_name),
                          stacklevel=2)
    return memory.cache(func, **kwargs)


class CacheMixin(object):
    """Mixin to add caching to a class.

    This class is a thin layer on top of joblib.Memory, that mainly adds a
    "caching level", similar to a "log level".

    Usage: to cache the results of a method, wrap it in self._cache()
    defined by this class. Caching is performed only if the user-specified
    cache level (self._memory_level) is greater than the value given as a
    parameter to self._cache(). See _cache() documentation for details.
    """

    def _cache(self, func, memory_level=1, **kwargs):
        """ Return a joblib.Memory object.

        The memory_level determines the level above which the wrapped
        function output is cached. By specifying a numeric value for
        this level, the user can to control the amount of cache memory
        used. This function will cache the function call or not
        depending on the cache level.

        Parameters
        ----------
        func: function
            The function the output of which is to be cached.

        memory_level: int
            The memory_level from which caching must be enabled for the wrapped
            function.

        Returns
        -------
        mem: joblib.Memory
            object that wraps the function func. This object may be
            a no-op, if the requested level is lower than the value given
            to _cache()). For consistency, a joblib.Memory object is always
            returned.
        """

        # Creates attributes if they don't exist
        # This is to make creating them in __init__() optional.
        if not hasattr(self, "memory_level"):
            self.memory_level = 0
        if not hasattr(self, "memory"):
            self.memory = Memory(cachedir=None)

        # If cache level is 0 but a memory object has been provided, set
        # memory_level to 1 with a warning.
        if self.memory_level == 0:
            if (isinstance(self.memory, basestring)
                    or self.memory.cachedir is not None):
                warnings.warn("memory_level is currently set to 0 but "
                              "a Memory object has been provided. "
                              "Setting memory_level to 1.")
                self.memory_level = 1
        verbose = getattr(self, 'verbose', 0)

        if self.memory_level < memory_level:
            mem = Memory(cachedir=None, verbose=verbose)
            return mem.cache(func, **kwargs)
        else:
            memory = self.memory
            if isinstance(memory, basestring):
                memory = Memory(cachedir=memory, verbose=verbose)
            if not isinstance(memory, memory_classes):
                raise TypeError("'memory' argument must be a string or a "
                                "joblib.Memory object.")
            if memory.cachedir is None:
                warnings.warn("Caching has been enabled (memory_level = %d) "
                              "but no Memory object or path has been provided"
                              " (parameter memory). Caching deactivated for "
                              "function %s." %
                              (self.memory_level, func.func_name))
            return memory.cache(func, **kwargs)


