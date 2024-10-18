"""Print the versions of python and several packages used in the project."""

import sys

import pkg_resources

DEPENDENCIES = [
    "joblib",
    "lxml",
    "matplotlib",
    "nibabel",
    "numpy",
    "pandas",
    "requests",
    "scipy",
    "scikit-learn",
]


def print_package_version(package_name, indent="  "):
    """Print install status and version of a package."""
    try:
        dist = pkg_resources.get_distribution(package_name)
    except pkg_resources.DistributionNotFound:
        provenance_info = "not installed"
    else:
        provenance_info = f"{dist.version} installed in {dist.location}"

    print(f"{indent}{package_name}: {provenance_info}")


if __name__ == "__main__":
    print("=" * 120)
    print(f"Python {sys.version!s}")
    print(f"from: {sys.executable}\n")

    print("Dependencies versions")
    for package_name in DEPENDENCIES:
        print_package_version(package_name)
    print("=" * 120)
