"""Testing package info"""

import pytest

import nibabel as nib
from nibabel.pkg_info import cmp_pkg_version


def test_pkg_info():
    """Smoke test nibabel.get_info()

    Hits:
        - nibabel.get_info
        - nibabel.pkg_info.get_pkg_info
        - nibabel.pkg_info.pkg_commit_hash
    """
    nib.get_info()


def test_version():
    # Test info version is the same as our own version
    assert nib.pkg_info.__version__ == nib.__version__


def test_cmp_pkg_version_0():
    # Test version comparator
    assert cmp_pkg_version(nib.__version__) == 0
    assert cmp_pkg_version('0.0') == -1
    assert cmp_pkg_version('1000.1000.1') == 1
    assert cmp_pkg_version(nib.__version__, nib.__version__) == 0

    # Check dev/RC sequence
    seq = ('3.0.0dev', '3.0.0rc1', '3.0.0rc1.post.dev', '3.0.0rc2', '3.0.0rc2.post.dev', '3.0.0')
    for stage1, stage2 in zip(seq[:-1], seq[1:]):
        assert cmp_pkg_version(stage1, stage2) == -1
        assert cmp_pkg_version(stage2, stage1) == 1


@pytest.mark.parametrize(
    ('test_ver', 'pkg_ver', 'exp_out'),
    [
        ('1.0', '1.0', 0),
        ('1.0.0', '1.0', 0),
        ('1.0', '1.0.0', 0),
        ('1.1', '1.1', 0),
        ('1.2', '1.1', 1),
        ('1.1', '1.2', -1),
        ('1.1.1', '1.1.1', 0),
        ('1.1.2', '1.1.1', 1),
        ('1.1.1', '1.1.2', -1),
        ('1.1', '1.1dev', 1),
        ('1.1dev', '1.1', -1),
        ('1.2.1', '1.2.1rc1', 1),
        ('1.2.1rc1', '1.2.1', -1),
        ('1.2.1rc1', '1.2.1rc', 1),
        ('1.2.1rc', '1.2.1rc1', -1),
        ('1.2.1b', '1.2.1a', 1),
        ('1.2.1a', '1.2.1b', -1),
        ('1.2.0+1', '1.2', 1),
        ('1.2', '1.2.0+1', -1),
        ('1.2.1+1', '1.2.1', 1),
        ('1.2.1', '1.2.1+1', -1),
        ('1.2.1rc1+1', '1.2.1', -1),
        ('1.2.1', '1.2.1rc1+1', 1),
        ('1.2.1rc1+1', '1.2.1+1', -1),
        ('1.2.1+1', '1.2.1rc1+1', 1),
    ],
)
def test_cmp_pkg_version_1(test_ver, pkg_ver, exp_out):
    # Test version comparator
    assert cmp_pkg_version(test_ver, pkg_ver) == exp_out


@pytest.mark.parametrize('args', [['foo.2'], ['foo.2', '1.0'], ['1.0', 'foo.2'], ['foo']])
def test_cmp_pkg_version_error(args):
    with pytest.raises(ValueError):
        cmp_pkg_version(*args)
