"""Utilities for testing nilearn."""
# Author: Alexandre Abraham, Philippe Gervais
# License: simplified BSD
import contextlib
import functools
import os
import sys
import tempfile
import warnings
import gc
import distutils
from pathlib import Path

import pytest
import sklearn

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
            pytest.skip('Test requires memory_profiler.')

        return dummy_func


    memory_usage = memory_used = None


def check_deprecation(func, match=None):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if distutils.version.LooseVersion(sklearn.__version__) < '0.22':
            with pytest.deprecated_call():
                result = func(*args, **kwargs)
        else:
            with pytest.warns(FutureWarning, match=match):
                result = func(*args, **kwargs)
        return result
    return wrapped


def assert_memory_less_than(memory_limit, tolerance,
                            callable_obj, *args, **kwargs):
    """Check memory consumption of a callable stays below a given limit.

    Parameters
    ----------
    memory_limit : int
        The expected memory limit in MiB.

    tolerance : float
        As memory_profiler results have some variability, this adds some
        tolerance around memory_limit. Accepted values are in range [0.0, 1.0].

    callable_obj : callable
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


def serialize_niimg(img, gzipped=True):
    """Serialize a Nifti1Image to nifti.

    Serialize to .nii.gz if gzipped, else to .nii Returns a `bytes` object.

    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        file_path = tmp_dir / "img.nii{}".format(".gz" if gzipped else "")
        img.to_filename(str(file_path))
        with file_path.open("rb") as f:
            return f.read()


@contextlib.contextmanager
def write_tmp_imgs(*imgs, **kwargs):
    """Context manager for writing Nifti images.

    Write nifti images in a temporary location, and remove them at the end of
    the block.

    Parameters
    ----------
    imgs : Nifti1Image
        Several Nifti images. Every format understood by nibabel.save is
        accepted.

    create_files : bool
        If True, imgs are written on disk and filenames are returned. If
        False, nothing is written, and imgs is returned as output. This is
        useful to test the two cases (filename / Nifti1Image) in the same
        loop.

    use_wildcards : bool
        If True, and create_files is True, imgs are written on disk and a
        matching glob is returned.

    Returns
    -------
    filenames : string or list of strings
        Filename(s) where input images have been written. If a single image
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
                    fd, filename = tempfile.mkstemp(prefix=prefix,
                                                    suffix=suffix,
                                                    dir=None)
                    os.close(fd)
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
            failures = []
            # Ensure all created files are removed
            for filename in filenames:
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    # ok, file already removed
                    pass
                except OSError as e:
                    # problem eg permission, or open file descriptor
                    failures.append(e)
            if failures:
                raise OSError(
                    "The following files could not be removed:\n".format(
                        "\n".join(str(e) for e in failures)
                    )
                )
    else:  # No-op
        if len(imgs) == 1:
            yield imgs[0]
        else:
            yield imgs


def are_tests_running():
    """Returns whether we are running the pytest test loader
    """
    return 'PYTEST_CURRENT_TEST' in os.environ


def skip_if_running_tests(msg=''):
    """ Raise a SkipTest if we appear to be running the pytest test loader.

    Parameters
    ----------
    msg : string, optional
        The message issued when a test is skipped.

    """
    if are_tests_running():
        pytest.skip(msg)
