"""
Test the _utils.cache_mixin module
"""
import glob
import json
import os
import shutil
import tempfile
from distutils.version import LooseVersion

import sklearn
from nose.tools import assert_false, assert_true, assert_equal
from sklearn.externals.joblib import Memory

import nilearn
from nilearn._utils import cache_mixin, CacheMixin
from nilearn._utils.testing import assert_raises_regex



def f(x):
    # A simple test function
    return x


def test_check_memory():
    # Test if _check_memory returns a memory object with the cachedir equal to
    # input path
    try:
        temp_dir = tempfile.mkdtemp()

        mem_none = Memory(cachedir=None)
        mem_temp = Memory(cachedir=temp_dir)

        for mem in [None, mem_none]:
            memory = cache_mixin._check_memory(mem, verbose=False)
            assert_true(memory, Memory)
            assert_equal(memory.cachedir, mem_none.cachedir)

        for mem in [temp_dir, mem_temp]:
            memory = cache_mixin._check_memory(mem, verbose=False)
            assert_equal(memory.cachedir, mem_temp.cachedir)
            assert_true(memory, Memory)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)



def test__safe_cache_dir_creation():
    # Test the _safe_cache function that is supposed to flush the
    # cache if the nibabel version changes
    try:
        temp_dir = tempfile.mkdtemp()
        mem = Memory(cachedir=temp_dir)
        version_file = os.path.join(temp_dir, 'joblib', 'module_versions.json')
        assert_false(os.path.exists(version_file))
        # First test that a version file get created
        cache_mixin._safe_cache(mem, f)
        assert_true(os.path.exists(version_file))
        # Test that it does not get recreated during the same session
        os.unlink(version_file)
        cache_mixin._safe_cache(mem, f)
        assert_false(os.path.exists(version_file))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test__safe_cache_flush():
    # Test the _safe_cache function that is supposed to flush the
    # cache if the nibabel version changes
    try:
        temp_dir = tempfile.mkdtemp()
        mem = Memory(cachedir=temp_dir)
        version_file = os.path.join(temp_dir, 'joblib', 'module_versions.json')
        # Create an mock version_file with old module versions
        with open(version_file, 'w') as f:
            json.dump({"nibabel": [0, 0]}, f)
        # Create some store structure
        nibabel_dir = os.path.join(temp_dir, 'joblib', 'nibabel_')
        os.makedirs(nibabel_dir)

        # First turn off version checking
        nilearn.CHECK_CACHE_VERSION = False
        cache_mixin._safe_cache(mem, f)
        assert_true(os.path.exists(nibabel_dir))

        # Second turn on version checking
        nilearn.CHECK_CACHE_VERSION = True
        # Make sure that the check will run again
        cache_mixin.__CACHE_CHECKED = {}
        with open(version_file, 'w') as f:
            json.dump({"nibabel": [0, 0]}, f)
        cache_mixin._safe_cache(mem, f)
        assert_true(os.path.exists(version_file))
        assert_false(os.path.exists(nibabel_dir))
    finally:
        pass
        # if os.path.exists(temp_dir):
        #    shutil.rmtree(temp_dir)


def test_cache_memory_level():
    temp_dir = tempfile.mkdtemp()
    job_glob = os.path.join(temp_dir, 'joblib', 'nilearn', 'tests',
                            'test_cache_mixin', 'f', '*')
    mem = Memory(cachedir=temp_dir, verbose=0)
    cache_mixin.cache(f, mem, func_memory_level=2, memory_level=1)(2)
    assert_equal(len(glob.glob(job_glob)), 0)
    cache_mixin.cache(f, Memory(cachedir=None))(2)
    assert_equal(len(glob.glob(job_glob)), 0)
    cache_mixin.cache(f, mem, func_memory_level=2, memory_level=3)(2)
    assert_equal(len(glob.glob(job_glob)), 2)
    cache_mixin.cache(f, mem)(3)
    assert_equal(len(glob.glob(job_glob)), 3)


class CacheMixinTest(CacheMixin):
    """Dummy mock object that wraps a CacheMixin."""

    def __init__(self, memory=None, memory_level=1):
        self.memory = memory
        self.memory_level = memory_level

    def run(self):
        self._cache(f)


def test_cache_mixin_with_expand_user():
    # Test the memory cache is correctly created when using ~.
    cache_dir = "~/nilearn_data/test_cache"
    expand_cache_dir = os.path.expanduser(cache_dir)
    mixin_mock = CacheMixinTest(cache_dir)

    try:
        assert_false(os.path.exists(expand_cache_dir))
        mixin_mock.run()
        assert_true(os.path.exists(expand_cache_dir))
    finally:
        if os.path.exists(expand_cache_dir):
            shutil.rmtree(expand_cache_dir)


def test_cache_mixin_without_expand_user():
    # Test the memory cache is correctly created when using ~.
    cache_dir = "~/nilearn_data/test_cache"
    expand_cache_dir = os.path.expanduser(cache_dir)
    mixin_mock = CacheMixinTest(cache_dir)

    try:
        assert_false(os.path.exists(expand_cache_dir))
        nilearn.EXPAND_PATH_WILDCARDS = False
        assert_raises_regex(ValueError,
                            "Given cache path parent directory doesn't",
                            mixin_mock.run)
        assert_false(os.path.exists(expand_cache_dir))
        nilearn.EXPAND_PATH_WILDCARDS = True
    finally:
        if os.path.exists(expand_cache_dir):
            shutil.rmtree(expand_cache_dir)


def test_cache_mixin_wrong_dirs():
    # Test the memory cache raises a ValueError when input base path doesn't
    # exist.

    for cache_dir in ("/bad_dir/cache",
                      "~/nilearn_data/tmp/test_cache"):
        expand_cache_dir = os.path.expanduser(cache_dir)
        mixin_mock = CacheMixinTest(cache_dir)

        try:
            assert_raises_regex(ValueError,
                                "Given cache path parent directory doesn't",
                                mixin_mock.run)
            assert_false(os.path.exists(expand_cache_dir))
        finally:
            if os.path.exists(expand_cache_dir):
                shutil.rmtree(expand_cache_dir)


def test_cache_shelving():
    try:
        temp_dir = tempfile.mkdtemp()
        job_glob = os.path.join(temp_dir, 'joblib', 'nilearn', 'tests',
                                'test_cache_mixin', 'f', '*')
        mem = Memory(cachedir=temp_dir, verbose=0)
        res = cache_mixin.cache(f, mem, shelve=True)(2)
        assert_equal(res.get(), 2)
        assert_equal(len(glob.glob(job_glob)), 1)
        res = cache_mixin.cache(f, mem, shelve=True)(2)
        assert_equal(res.get(), 2)
        assert_equal(len(glob.glob(job_glob)), 1)
    finally:
        del mem
        shutil.rmtree(temp_dir, ignore_errors=True)
