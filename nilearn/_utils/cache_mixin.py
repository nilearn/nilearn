"""
Mixin for cache with joblib
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import json
import warnings
import os
import shutil
from distutils.version import LooseVersion

import nibabel
import sklearn

from sklearn.externals.joblib import Memory

MEMORY_CLASSES = (Memory, )

try:
    from joblib import Memory as JoblibMemory
    MEMORY_CLASSES = (Memory, JoblibMemory)
except ImportError:
    pass

import nilearn

from .compat import _basestring

__CACHE_CHECKED = dict()


def _check_memory(memory, verbose=0):
    """Function to ensure an instance of a joblib.Memory object.

    Parameters
    ----------
    memory: None or instance of joblib.Memory or str
        Used to cache the masking process.
        If a str is given, it is the path to the caching directory.

    verbose : int, optional (default 0)
        Verbosity level.

    Returns
    -------
    instance of joblib.Memory.
    """
    if memory is None:
        memory = Memory(cachedir=None, verbose=verbose)
    if isinstance(memory, _basestring):
        cache_dir = memory
        if nilearn.EXPAND_PATH_WILDCARDS:
            cache_dir = os.path.expanduser(cache_dir)

        # Perform some verifications on given path.
        split_cache_dir = os.path.split(cache_dir)
        if (len(split_cache_dir) > 1 and
                (not os.path.exists(split_cache_dir[0]) and
                    split_cache_dir[0] != '')):
            if (not nilearn.EXPAND_PATH_WILDCARDS and
                    cache_dir.startswith("~")):
                # Maybe the user want to enable expanded user path.
                error_msg = ("Given cache path parent directory doesn't "
                             "exists, you gave '{0}'. Enabling "
                             "nilearn.EXPAND_PATH_WILDCARDS could solve "
                             "this issue.".format(split_cache_dir[0]))
            elif memory.startswith("~"):
                # Path built on top of expanded user path doesn't exist.
                error_msg = ("Given cache path parent directory doesn't "
                             "exists, you gave '{0}' which was expanded "
                             "as '{1}' but doesn't exist either. Use "
                             "nilearn.EXPAND_PATH_WILDCARDS to deactivate "
                             "auto expand user path (~) behavior."
                             .format(split_cache_dir[0],
                                     os.path.dirname(memory)))
            else:
                # The given cache base path doesn't exist.
                error_msg = ("Given cache path parent directory doesn't "
                             "exists, you gave '{0}'."
                             .format(split_cache_dir[0]))
            raise ValueError(error_msg)

        memory = Memory(cachedir=cache_dir, verbose=verbose)
    return memory


def _safe_cache(memory, func, **kwargs):
    """ A wrapper for mem.cache that flushes the cache if the version
        number of nibabel has changed.
    """
    cachedir = memory.cachedir

    if cachedir is None or cachedir in __CACHE_CHECKED:
        return memory.cache(func, **kwargs)

    version_file = os.path.join(cachedir, 'module_versions.json')

    versions = dict()
    if os.path.exists(version_file):
        with open(version_file, 'r') as _version_file:
            versions = json.load(_version_file)

    modules = (nibabel, )
    # Keep only the major + minor version numbers
    my_versions = dict((m.__name__, LooseVersion(m.__version__).version[:2])
                       for m in modules)
    commons = set(versions.keys()).intersection(set(my_versions.keys()))
    collisions = [m for m in commons if versions[m] != my_versions[m]]

    # Flush cache if version collision
    if len(collisions) > 0:
        if nilearn.CHECK_CACHE_VERSION:
            warnings.warn("Incompatible cache in %s: "
                          "different version of nibabel. Deleting "
                          "the cache. Put nilearn.CHECK_CACHE_VERSION "
                          "to false to avoid this behavior."
                          % cachedir)
            try:
                tmp_dir = (os.path.split(cachedir)[:-1]
                           + ('old_%i' % os.getpid(), ))
                tmp_dir = os.path.join(*tmp_dir)
                # We use rename + unlink to be more robust to race
                # conditions
                os.rename(cachedir, tmp_dir)
                shutil.rmtree(tmp_dir)
            except OSError:
                # Another process could have removed this dir
                pass

            try:
                os.makedirs(cachedir)
            except OSError:
                # File exists?
                pass
        else:
            warnings.warn("Incompatible cache in %s: "
                          "old version of nibabel." % cachedir)

    # Write json files if configuration is different
    if versions != my_versions:
        with open(version_file, 'w') as _version_file:
            json.dump(my_versions, _version_file)

    __CACHE_CHECKED[cachedir] = True

    return memory.cache(func, **kwargs)


class _ShelvedFunc(object):
    """Work around for Python 2, for which pickle fails on instance method"""
    def __init__(self, func):
        self.func = func
        self.func_name = func.__name__ + '_shelved'

    def __call__(self, *args, **kwargs):
            return self.func.call_and_shelve(*args, **kwargs)


def cache(func, memory, func_memory_level=None, memory_level=None,
          shelve=False, **kwargs):
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

    func_memory_level: int, optional
        The memory_level from which caching must be enabled for the wrapped
        function.

    memory_level: int, optional
        The memory_level used to determine if function call must
        be cached or not (if user_memory_level is equal of greater than
        func_memory_level the function is cached)

    shelve: bool
        Whether to return a joblib MemorizedResult, callable by a .get()
        method, instead of the return value of func

    kwargs: keyword arguments
        The keyword arguments passed to memory.cache

    Returns
    -------
    mem: joblib.MemorizedFunc, wrapped in _ShelvedFunc if shelving
        Object that wraps the function func to cache its further call.
        This object may be a no-op, if the requested level is lower
        than the value given to _cache()).
        For consistency, a callable object is always returned.
    """
    verbose = kwargs.get('verbose', 0)

    # memory_level and func_memory_level must be both None or both integers.
    memory_levels = [memory_level, func_memory_level]
    both_params_integers = all(isinstance(lvl, int) for lvl in memory_levels)
    both_params_none = all(lvl is None for lvl in memory_levels)

    if not (both_params_integers or both_params_none):
        raise ValueError('Reference and user memory levels must be both None '
                         'or both integers.')

    if memory is not None and (func_memory_level is None or
                               memory_level >= func_memory_level):
        if isinstance(memory, _basestring):
            memory = Memory(cachedir=memory, verbose=verbose)
        if not isinstance(memory, MEMORY_CLASSES):
            raise TypeError("'memory' argument must be a string or a "
                            "joblib.Memory object. "
                            "%s %s was given." % (memory, type(memory)))
        if (memory.cachedir is None and memory_level is not None
                and memory_level > 1):
            warnings.warn("Caching has been enabled (memory_level = %d) "
                          "but no Memory object or path has been provided"
                          " (parameter memory). Caching deactivated for "
                          "function %s." %
                          (memory_level, func.__name__),
                          stacklevel=2)
    else:
        memory = Memory(cachedir=None, verbose=verbose)
    cached_func = _safe_cache(memory, func, **kwargs)
    if shelve:
        cached_func = _ShelvedFunc(cached_func)
    return cached_func


class CacheMixin(object):
    """Mixin to add caching to a class.

    This class is a thin layer on top of joblib.Memory, that mainly adds a
    "caching level", similar to a "log level".

    Usage: to cache the results of a method, wrap it in self._cache()
    defined by this class. Caching is performed only if the user-specified
    cache level (self._memory_level) is greater than the value given as a
    parameter to self._cache(). See _cache() documentation for details.
    """
    def _cache(self, func, func_memory_level=1, shelve=False, **kwargs):
        """Return a joblib.Memory object.

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

        shelve: bool
            Whether to return a joblib MemorizedResult, callable by a .get()
            method, instead of the return value of func

        Returns
        -------
        mem: joblib.MemorizedFunc, wrapped in _ShelvedFunc if shelving
            Object that wraps the function func to cache its further call.
            This object may be a no-op, if the requested level is lower
            than the value given to _cache()).
            For consistency, a callable object is always returned.
        """
        verbose = getattr(self, 'verbose', 0)

        # Creates attributes if they don't exist
        # This is to make creating them in __init__() optional.
        if not hasattr(self, "memory_level"):
            self.memory_level = 0
        if not hasattr(self, "memory"):
            self.memory = Memory(cachedir=None, verbose=verbose)
        self.memory = _check_memory(self.memory, verbose=verbose)

        # If cache level is 0 but a memory object has been provided, set
        # memory_level to 1 with a warning.
        if self.memory_level == 0 and self.memory.cachedir is not None:
            warnings.warn("memory_level is currently set to 0 but "
                          "a Memory object has been provided. "
                          "Setting memory_level to 1.")
            self.memory_level = 1

        return cache(func, self.memory, func_memory_level=func_memory_level,
                     memory_level=self.memory_level, shelve=shelve,
                     **kwargs)
