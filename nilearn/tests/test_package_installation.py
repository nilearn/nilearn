from nilearn.version import _compare_version

_NILEARN_INSTALL_MSG = "See %s for installation information." % (
    "https://nilearn.github.io/stable/introduction.html#installation"
)

REQUIRED_MODULE_METADATA = (
    (
        "numpy",
        {
            "min_version": "1.19",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    (
        "scipy",
        {
            "min_version": "1.6",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    (
        "sklearn",
        {
            "min_version": "1.0.0",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    (
        "joblib",
        {
            "min_version": "1.0.0",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    ("nibabel", {"min_version": "3.2", "required_at_installation": False}),
    (
        "pandas",
        {
            "min_version": "1.1.5",
            "required_at_installation": True,
            "install_info": _NILEARN_INSTALL_MSG,
        },
    ),
    ("requests", {"min_version": "2.25", "required_at_installation": False}),
)


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
