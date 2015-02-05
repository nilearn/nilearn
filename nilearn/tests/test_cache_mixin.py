"""
Test the _utils.cache_mixin module
"""
import os
import shutil
import tempfile
import json
import glob

from nose.tools import assert_false, assert_true

from sklearn.externals.joblib import Memory

import nilearn
from .._utils import cache_mixin


def f(x):
    # A simple test function
    return x


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
        nilearn.check_cache_version = False
        cache_mixin._safe_cache(mem, f)
        assert_true(os.path.exists(nibabel_dir))

        # Second turn on version checking
        nilearn.check_cache_version = True
        # Make sure that the check will run again
        cache_mixin.__cache_checked = {}
        with open(version_file, 'w') as f:
            json.dump({"nibabel": [0, 0]}, f)
        cache_mixin._safe_cache(mem, f)
        assert_true(os.path.exists(version_file))
        assert_false(os.path.exists(nibabel_dir))
    finally:
        pass
        #if os.path.exists(temp_dir):
        #    shutil.rmtree(temp_dir)


def test_cache_memory_level():
    temp_dir = tempfile.mkdtemp()
    job_glob = os.path.join(temp_dir, 'joblib', '*')
    mem = Memory(cachedir=temp_dir, verbose=0)
    cache_mixin.cache(f, mem)(2)
    assert_true(len(glob.glob(job_glob)) == 0)
    cache_mixin.cache(f, mem, func_memory_level=2, memory_level=1)(2)
    assert_true(len(glob.glob(job_glob)) == 0)
    cache_mixin.cache(f, mem, func_memory_level=2, memory_level=3)(2)
    assert_true(len(glob.glob(job_glob)) == 2)
