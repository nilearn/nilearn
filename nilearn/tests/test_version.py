import pytest
from nilearn.version import _compare_version


@pytest.mark.parametrize("version_a,operator,version_b",
                         [('0.1.0', '>', '0.0.1'),
                          ('0.1.0', '>=', '0.0.1'),
                          ('0.1', '==', '0.1.0'),
                          ('0.0.0', '<', '0.1.0'),
                          ('1.0', '!=', '0.1.0')])
def test_compare_version(version_a, operator, version_b):
    assert _compare_version(version_a, operator, version_b)


def test_compare_version_error():
    with pytest.raises(ValueError,
                       match=("'_compare_version' received an "
                              "unexpected operator <>.")):
        _compare_version('0.1.0', '<>', '1.1.0')
