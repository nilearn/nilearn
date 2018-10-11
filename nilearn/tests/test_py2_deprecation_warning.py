from distutils.version import LooseVersion
import warnings
import sys

from nose.tools import assert_true
from nilearn import _py2_deprecation_warning


def test_py2_deprecation_warning():
    with warnings.catch_warnings(record=True) as raised_warnings:
        _py2_deprecation_warning()
        assert_true(raised_warnings[0].category is DeprecationWarning)


# def test_py2_warning_at_nilearn_import():
#     with warnings.catch_warnings(record=True) as raised_warnings:
#         warnings.simplefilter('always')
#         from nilearn.plotting import view_stat_map
#         if LooseVersion(sys.version) < LooseVersion('3'):
#             assert_true(raised_warnings[0].category is DeprecationWarning)
#         else:
#             assert raised_warnings is None


if __name__ == '__main__':
    test_py2_deprecation_warning()
    # test_py2_warning_at_nilearn_import()
