import sys
import warnings

from nose.tools import assert_true

# import time warnings don't interfere with warning's tests
with warnings.catch_warnings(record=True):
    from nilearn import _py34_deprecation_warning
    from nilearn import _py2_deprecation_warning
    from nilearn import _python_deprecation_warnings


def test_py2_deprecation_warning():
    with warnings.catch_warnings(record=True) as raised_warnings:
            _py2_deprecation_warning()
    assert_true(raised_warnings[0].category is DeprecationWarning)
    assert_true(
            str(raised_warnings[0].message).startswith(
                    'Python2 support is deprecated')
            )


def test_py34_deprecation_warning():
    with warnings.catch_warnings(record=True) as raised_warnings:
        _py34_deprecation_warning()
    assert_true(raised_warnings[0].category is DeprecationWarning)
    assert_true(
            str(raised_warnings[0].message).startswith(
            'Python 3.4 support is deprecated')
            )


def test_python_deprecation_warnings():
    with warnings.catch_warnings(record=True) as raised_warnings:
        _python_deprecation_warnings()
    if sys.version_info.major == 2:
        assert_true(raised_warnings[0].category is DeprecationWarning)
        assert_true(
                str(raised_warnings[0].message).startswith(
                        'Python2 support is deprecated')
                )
    elif sys.version_info.major == 3 and sys.version_info.minor == 4:
        assert_true(raised_warnings[0].category is DeprecationWarning)
        assert_true(
                str(raised_warnings[0].message).startswith(
                        'Python 3.4 support is deprecated')
                )
    else:
        assert_true(len(raised_warnings) == 0)


def test_warnings_filter_scope():
    """
    Tests that warnings generated at Nilearn import in Python 2, 3.4 envs
    do not change the warnings filter for subsequent warnings.
    """
    with warnings.catch_warnings(record=True) as raised_warnings:
        warnings.warn('Dummy warning 1')  # Will be raised.
        warnings.filterwarnings("ignore", message="Dummy warning")
        warnings.warn('Dummy warning 2')  # Will not be raised.
        import nilearn  # Irrespective of warning raised in py2, py3.4 ...
        warnings.warn('Dummy warning 3')  # ...this should not be raised.
    assert str(raised_warnings[0].message) == 'Dummy warning 1'
    assert str(raised_warnings[-1].message) != 'Dummy warning 3'
