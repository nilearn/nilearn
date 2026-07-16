"""Print the versions of Python and several packages used in the project."""

import importlib.metadata
import sys

from rich import print

DEPENDENCIES = [
    "joblib",
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
        version = importlib.metadata.version(package_name)
        provenance_info = f"{version} installed"
    except importlib.metadata.PackageNotFoundError:
        provenance_info = "not installed"

    print(f"{indent}{package_name}: {provenance_info}")


if __name__ == "__main__":
    print("=" * 120)
    print(f"Python {sys.version!s}")
    print(f"from: {sys.executable}\n")

    print("Dependencies versions")
    for package_name in DEPENDENCIES:
        print_package_version(package_name)
    print("=" * 120)
