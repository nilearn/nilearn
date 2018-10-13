"""Utilities for testing nilearn."""
# Author: Alexandre Abraham, Philippe Gervais
# License: simplified BSD
import contextlib
import functools
import inspect
import os
import sys
import tempfile
import warnings
import gc

import numpy as np
from sklearn.utils.testing import assert_warns

from .compat import _basestring, _urllib
from ..datasets.utils import _fetch_files


try:
    from nose.tools import assert_raises_regex
except ImportError:
    # For Py 2.7
    from nose.tools import assert_raises_regexp as assert_raises_regex


# we use memory_profiler library for memory consumption checks
try:
    from memory_profiler import memory_usage

    def with_memory_profiler(func):
        """A decorator to skip tests requiring memory_profiler."""
        return func

    def memory_used(func, *args, **kwargs):
        """Compute memory usage when executing func."""
        def func_3_times(*args, **kwargs):
            for _ in range(3):
                func(*args, **kwargs)

        gc.collect()
        mem_use = memory_usage((func_3_times, args, kwargs), interval=0.001)
        return max(mem_use) - min(mem_use)

except ImportError:
    def with_memory_profiler(func):
        """A decorator to skip tests requiring memory_profiler."""
        def dummy_func():
            import nose
            raise nose.SkipTest('Test requires memory_profiler.')
        return dummy_func

    memory_usage = memory_used = None


def assert_memory_less_than(memory_limit, tolerance,
                            callable_obj, *args, **kwargs):
    """Check memory consumption of a callable stays below a given limit.

    Parameters
    ----------
    memory_limit : int
        The expected memory limit in MiB.
    tolerance: float
        As memory_profiler results have some variability, this adds some
        tolerance around memory_limit. Accepted values are in range [0.0, 1.0].
    callable_obj: callable
        The function to be called to check memory consumption.

    """
    mem_used = memory_used(callable_obj, *args, **kwargs)

    if mem_used > memory_limit * (1 + tolerance):
        raise ValueError("Memory consumption measured ({0:.2f} MiB) is "
                         "greater than required memory limit ({1} MiB) within "
                         "accepted tolerance ({2:.2f}%)."
                         "".format(mem_used, memory_limit, tolerance * 100))

    # We are confident in memory_profiler measures above 100MiB.
    # We raise an error if the measure is below the limit of 50MiB to avoid
    # false positive.
    if mem_used < 50:
        raise ValueError("Memory profiler measured an untrustable memory "
                         "consumption ({0:.2f} MiB). The expected memory "
                         "limit was {1:.2f} MiB. Try to bench with larger "
                         "objects (at least 100MiB in memory).".
                         format(mem_used, memory_limit))


class MockRequest(object):
    def __init__(self, url):
        self.url = url

    def add_header(*args):
        pass


class MockOpener(object):
    def __init__(self):
        pass

    def open(self, request):
        return request.url


@contextlib.contextmanager
def write_tmp_imgs(*imgs, **kwargs):
    """Context manager for writing Nifti images.

    Write nifti images in a temporary location, and remove them at the end of
    the block.

    Parameters
    ----------
    imgs: Nifti1Image
        Several Nifti images. Every format understood by nibabel.save is
        accepted.

    create_files: bool
        if True, imgs are written on disk and filenames are returned. If
        False, nothing is written, and imgs is returned as output. This is
        useful to test the two cases (filename / Nifti1Image) in the same
        loop.

    use_wildcards: bool
        if True, and create_files is True, imgs are written on disk and a
        matching glob is returned.

    Returns
    -------
    filenames: string or list of
        filename(s) where input images have been written. If a single image
        has been given as input, a single string is returned. Otherwise, a
        list of string is returned.
    """
    valid_keys = set(("create_files", "use_wildcards"))
    input_keys = set(kwargs.keys())
    invalid_keys = input_keys - valid_keys
    if len(invalid_keys) > 0:
        raise TypeError("%s: unexpected keyword argument(s): %s" %
                        (sys._getframe().f_code.co_name,
                         " ".join(invalid_keys)))
    create_files = kwargs.get("create_files", True)
    use_wildcards = kwargs.get("use_wildcards", False)

    prefix = "nilearn_"
    suffix = ".nii"

    if create_files:
        filenames = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for img in imgs:
                    filename = tempfile.mktemp(prefix=prefix,
                                               suffix=suffix,
                                               dir=None)
                    filenames.append(filename)
                    img.to_filename(filename)
                    del img

                if use_wildcards:
                    yield prefix + "*" + suffix
                else:
                    if len(imgs) == 1:
                        yield filenames[0]
                    else:
                        yield filenames
        finally:
            # Ensure all created files are removed
            for filename in filenames:
                os.remove(filename)
    else:  # No-op
        if len(imgs) == 1:
            yield imgs[0]
        else:
            yield imgs


class mock_request(object):
    def __init__(self):
        """Object that mocks the urllib (future) module to store downloaded filenames.

        `urls` is the list of the files whose download has been
        requested.
        """
        self.urls = set()

    def reset(self):
        self.urls = set()

    def Request(self, url):
        self.urls.add(url)
        return MockRequest(url)

    def build_opener(self, *args, **kwargs):
        return MockOpener()


def wrap_chunk_read_(_chunk_read_):
    def mock_chunk_read_(response, local_file, initial_size=0, chunk_size=8192,
                         report_hook=None, verbose=0):
        if not isinstance(response, _basestring):
            return _chunk_read_(response, local_file,
                                initial_size=initial_size,
                                chunk_size=chunk_size,
                                report_hook=report_hook, verbose=verbose)
        return response
    return mock_chunk_read_


def mock_chunk_read_raise_error_(response, local_file, initial_size=0,
                                 chunk_size=8192, report_hook=None,
                                 verbose=0):
    raise _urllib.errors.HTTPError("url", 418, "I'm a teapot", None, None)


class FetchFilesMock (object):
    _mock_fetch_files = functools.partial(_fetch_files, mock=True)

    def __init__(self):
        """Create a mock that can fill a CSV file if needed
        """
        self.csv_files = {}

    def add_csv(self, filename, content):
        self.csv_files[filename] = content

    def __call__(self, *args, **kwargs):
        """Load requested dataset, downloading it if needed or requested.

        For test purpose, instead of actually fetching the dataset, this
        function creates empty files and return their paths.
        """
        filenames = self._mock_fetch_files(*args, **kwargs)
        # Fill CSV files with given content if needed
        for fname in filenames:
            basename = os.path.basename(fname)
            if basename in self.csv_files:
                array = self.csv_files[basename]

                # np.savetxt does not have a header argument for numpy 1.6
                # np.savetxt(fname, array, delimiter=',', fmt="%s",
                #            header=','.join(array.dtype.names))
                # We need to add the header ourselves
                with open(fname, 'wb') as f:
                    header = '# {0}\n'.format(','.join(array.dtype.names))
                    f.write(header.encode())
                    np.savetxt(f, array, delimiter=',', fmt='%s')

        return filenames



def is_nose_running():
    """Returns whether we are running the nose test loader
    """
    if 'nose' not in sys.modules:
        return
    try:
        import nose
    except ImportError:
        return False
    # Now check that we have the loader in the call stask
    stack = inspect.stack()
    loader_file_name = nose.loader.__file__
    if loader_file_name.endswith('.pyc'):
        loader_file_name = loader_file_name[:-1]
    for _, file_name, _, _, _, _ in stack:
        if file_name == loader_file_name:
            return True
    return False


def skip_if_running_nose(msg=''):
    """ Raise a SkipTest if we appear to be running the nose test loader.

    Parameters
    ----------
    msg: string, optional
        The message issued when SkipTest is raised
    """
    if is_nose_running():
        import nose
        raise nose.SkipTest(msg)


# Backport: On some nose versions, assert_less_equal is not present
try:
    from nose.tools import assert_less_equal
except ImportError:
    def assert_less_equal(a, b):
        if a > b:
            raise AssertionError("%f is not less or equal than %f" % (a, b))

try:
    from nose.tools import assert_less
except ImportError:
    def assert_less(a, b):
        if a >= b:
            raise AssertionError("%f is not less than %f" % (a, b))
