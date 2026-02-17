"""Testing optpkg module"""

import builtins
import sys
import types
from unittest import SkipTest, mock

import pytest
from packaging.version import Version

from nibabel.optpkg import optional_package
from nibabel.tripwire import TripWire, TripWireError


def assert_good(pkg_name, min_version=None):
    pkg, have_pkg, setup = optional_package(pkg_name, min_version=min_version)
    assert have_pkg
    assert sys.modules[pkg_name] == pkg
    assert setup() is None


def assert_bad(pkg_name, min_version=None):
    pkg, have_pkg, setup = optional_package(pkg_name, min_version=min_version)
    assert not have_pkg
    assert isinstance(pkg, TripWire)
    with pytest.raises(TripWireError):
        pkg.a_method
    with pytest.raises(SkipTest):
        setup()


def test_basic():
    # We always have os
    assert_good('os')
    # Subpackage
    assert_good('os.path')
    # We never have package _not_a_package
    assert_bad('_not_a_package')

    # Only disrupt imports for "nottriedbefore" package
    orig_import = builtins.__import__

    def raise_Exception(*args, **kwargs):
        if args[0] == 'nottriedbefore':
            raise Exception(
                'non ImportError could be thrown by some malfunctioning module '
                'upon import, and optional_package should catch it too'
            )
        return orig_import(*args, **kwargs)

    with mock.patch.object(builtins, '__import__', side_effect=raise_Exception):
        assert_bad('nottriedbefore')


def test_versions():
    fake_name = '_a_fake_package'
    fake_pkg = types.ModuleType(fake_name)
    assert 'fake_pkg' not in sys.modules
    # Not inserted yet
    assert_bad(fake_name)
    try:
        sys.modules[fake_name] = fake_pkg
        # No __version__ yet
        assert_good(fake_name)  # With no version check
        assert_bad(fake_name, '1.0')
        # We can make an arbitrary callable to check version
        assert_good(fake_name, lambda pkg: True)
        # Now add a version
        fake_pkg.__version__ = '2.0'
        # We have fake_pkg > 1.0
        for min_ver in (None, '1.0', Version('1.0'), lambda pkg: True):
            assert_good(fake_name, min_ver)
        # We never have fake_pkg > 100.0
        for min_ver in ('100.0', Version('100.0'), lambda pkg: False):
            assert_bad(fake_name, min_ver)
        # Check error string for bad version
        pkg, _, _ = optional_package(fake_name, min_version='3.0')
        try:
            pkg.some_method
        except TripWireError as err:
            assert str(err) == 'These functions need _a_fake_package version >= 3.0'
    finally:
        del sys.modules[fake_name]
