"""Test the _utils.cache_mixin module."""

import shutil
from pathlib import Path

import pytest
from joblib import Memory

import nilearn
from nilearn._utils import CacheMixin, cache_mixin


def _get_subdirs(top_dir):
    top_dir = Path(top_dir)
    children = list(top_dir.glob("*"))
    return [child for child in children if child.is_dir()]


def f(x):
    # A simple test function
    return x


def test_check_memory(tmp_path):
    # Test if _check_memory returns a memory object with the location equal to
    # input path

    mem_none = Memory(location=None)
    mem_temp = Memory(location=str(tmp_path))

    for mem in [None, mem_none]:
        memory = cache_mixin._check_memory(mem, verbose=0)
        assert memory, Memory
        assert memory.location == mem_none.location

    for mem in [str(tmp_path), mem_temp]:
        memory = cache_mixin._check_memory(mem, verbose=0)
        assert memory.location == mem_temp.location
        assert memory, Memory


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
    expand_cache_dir = Path(cache_dir).expanduser()
    mixin_mock = CacheMixinTest(cache_dir)

    try:
        assert not expand_cache_dir.exists()
        mixin_mock.run()
        assert expand_cache_dir.exists()
    finally:
        if expand_cache_dir.exists():
            shutil.rmtree(expand_cache_dir)


def test_cache_mixin_without_expand_user():
    # Test the memory cache is correctly created when using ~.
    cache_dir = "~/nilearn_data/test_cache"
    expand_cache_dir = Path(cache_dir).expanduser()
    mixin_mock = CacheMixinTest(cache_dir)

    try:
        assert not expand_cache_dir.exists()
        nilearn.EXPAND_PATH_WILDCARDS = False
        with pytest.raises(
            ValueError, match="Given cache path parent directory doesn't"
        ):
            mixin_mock.run()
        assert not expand_cache_dir.exists()
        nilearn.EXPAND_PATH_WILDCARDS = True
    finally:
        if expand_cache_dir.exists():
            shutil.rmtree(expand_cache_dir)


def test_cache_mixin_wrong_dirs():
    # Test the memory cache raises a ValueError when input base path doesn't
    # exist.

    for cache_dir in ("/bad_dir/cache", "~/nilearn_data/tmp/test_cache"):
        expand_cache_dir = Path(cache_dir).expanduser()
        mixin_mock = CacheMixinTest(cache_dir)

        try:
            with pytest.raises(
                ValueError, match="Given cache path parent directory doesn't"
            ):
                mixin_mock.run()
            assert not expand_cache_dir.exists()
        finally:
            if expand_cache_dir.exists():
                shutil.rmtree(expand_cache_dir)


def test_cache_memory_level(tmp_path):
    joblib_dir = (
        tmp_path
        / "joblib"
        / "nilearn"
        / "_utils"
        / "tests"
        / "test_cache_mixin"
        / "f"
    )

    cache_mixin.cache(f, Memory(location=None))(2)
    assert len(_get_subdirs(joblib_dir)) == 0

    mem = Memory(location=str(tmp_path), verbose=0)

    cache_mixin.cache(f, mem, func_memory_level=2, memory_level=1)(2)
    assert len(_get_subdirs(joblib_dir)) == 0

    cache_mixin.cache(f, mem, func_memory_level=2, memory_level=3)(2)
    assert len(_get_subdirs(joblib_dir)) == 1

    cache_mixin.cache(f, mem)(3)
    assert len(_get_subdirs(joblib_dir)) == 2


def test_cache_shelving(tmp_path):
    joblib_dir = (
        tmp_path
        / "joblib"
        / "nilearn"
        / "_utils"
        / "tests"
        / "test_cache_mixin"
        / "f"
    )
    mem = Memory(location=str(tmp_path), verbose=0)
    res = cache_mixin.cache(f, mem, shelve=True)(2)
    assert res.get() == 2
    assert len(_get_subdirs(joblib_dir)) == 1
    res = cache_mixin.cache(f, mem, shelve=True)(2)
    assert res.get() == 2
    assert len(_get_subdirs(joblib_dir)) == 1
