from distutils.version import LooseVersion

from nilearn.version import REQUIRED_MODULE_METADATA


def test_required_package_installation():
    for package_specs in REQUIRED_MODULE_METADATA:
        package = package_specs[0]
        min_version = package_specs[1]['min_version']
        imported_package = __import__(package)
        installed_version = imported_package.__version__
        assert LooseVersion(installed_version) >= LooseVersion(min_version)
        print(package, 'min:', min_version, 'installed:', installed_version)


if __name__ == '__main__':
    test_required_package_installation()
