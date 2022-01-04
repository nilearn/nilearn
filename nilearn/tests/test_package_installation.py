import pytest
from nilearn.version import (
    REQUIRED_MODULE_METADATA, _compare_version
)


@pytest.mark.parametrize("version_a,operator,version_b",
                         [('0.1.0', '>', '0.0.1'),
                          ('0.1.0', '>=', '0.0.1'),
                          ('0.1', '==', '0.1.0'),
                          ('0.0.0', '<', '0.1.0'),
                          ('1.0', '!=', '0.1.0')])
def test_compare_version(version_a, operator, version_b):
    assert _compare_version(version_a, operator, version_b)


def test_required_package_installation():
    for package_specs in REQUIRED_MODULE_METADATA:
        package = package_specs[0]
        min_version = package_specs[1]['min_version']
        imported_package = __import__(package)
        installed_version = imported_package.__version__
        assert _compare_version(installed_version, '>=', min_version)
        print(package, 'min:', min_version, 'installed:', installed_version)


if __name__ == '__main__':
    test_required_package_installation()
