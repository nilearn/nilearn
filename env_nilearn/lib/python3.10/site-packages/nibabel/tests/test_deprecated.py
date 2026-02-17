"""Testing `deprecated` module"""

import warnings

import pytest

from nibabel import pkg_info
from nibabel.deprecated import (
    FutureWarningMixin,
    ModuleProxy,
    alert_future_error,
    deprecate_with_version,
)
from nibabel.tests.test_deprecator import TestDeprecatorFunc as _TestDF


def setup_module():
    # Hack nibabel version string
    pkg_info.cmp_pkg_version.__defaults__ = ('2.0',)


def teardown_module():
    # Hack nibabel version string back again
    pkg_info.cmp_pkg_version.__defaults__ = (pkg_info.__version__,)


def test_module_proxy():
    # Test proxy for module
    mp = ModuleProxy('nibabel.deprecated')
    assert hasattr(mp, 'ModuleProxy')
    assert mp.ModuleProxy is ModuleProxy
    assert repr(mp) == '<module proxy for nibabel.deprecated>'


def test_futurewarning_mixin():
    # Test mixin for FutureWarning
    class C:
        def __init__(self, val):
            self.val = val

        def meth(self):
            return self.val

    class D(FutureWarningMixin, C):
        pass

    class E(FutureWarningMixin, C):
        warn_message = 'Oh no, not this one'

    with warnings.catch_warnings(record=True) as warns:
        c = C(42)
        assert c.meth() == 42
        assert warns == []
        d = D(42)
        assert d.meth() == 42
        warn = warns.pop(0)
        assert warn.category == FutureWarning
        assert str(warn.message) == 'This class will be removed in future versions'
        e = E(42)
        assert e.meth() == 42
        warn = warns.pop(0)
        assert warn.category == FutureWarning
        assert str(warn.message) == 'Oh no, not this one'


class TestNibabelDeprecator(_TestDF):
    """Test deprecations against nibabel version"""

    dep_func = deprecate_with_version


def test_dev_version():
    # Test that a dev version doesn't trigger deprecation error

    @deprecate_with_version('foo', until='2.0')
    def func():
        return 99

    try:
        pkg_info.cmp_pkg_version.__defaults__ = ('2.0dev',)
        # No error, even though version is dev version of current
        with pytest.deprecated_call():
            assert func() == 99
    finally:
        pkg_info.cmp_pkg_version.__defaults__ = ('2.0',)


def test_alert_future_error():
    with pytest.warns(FutureWarning):
        alert_future_error(
            'Message',
            '9999.9.9',
            warning_rec='Silence this warning by doing XYZ.',
            error_rec='Fix this issue by doing XYZ.',
        )
    with pytest.raises(RuntimeError):
        alert_future_error(
            'Message',
            '1.0.0',
            warning_rec='Silence this warning by doing XYZ.',
            error_rec='Fix this issue by doing XYZ.',
        )
    with pytest.raises(ValueError):
        alert_future_error(
            'Message',
            '1.0.0',
            warning_rec='Silence this warning by doing XYZ.',
            error_rec='Fix this issue by doing XYZ.',
            error_class=ValueError,
        )
    with pytest.raises(ValueError):
        alert_future_error(
            'Message',
            '2.0.0',  # Error if we equal the (patched) version
            warning_rec='Silence this warning by doing XYZ.',
            error_rec='Fix this issue by doing XYZ.',
            error_class=ValueError,
        )
