import sys
import pkg_resources

DEPENDENCIES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "nibabel",
]


def print_package_version(package_name, indent="  "):
    try:
        dist = pkg_resources.get_distribution(package_name)
    except pkg_resources.DistributionNotFound:
        provenance_info = "not installed"
    else:
        provenance_info = f"{dist.version} installed in {dist.location}"

    print(f"{indent}{package_name}: {provenance_info}")


if __name__ == "__main__":
    print("=" * 120)
    print(f"Python {str(sys.version)}")
    print("from: %s\n" % sys.executable)

    print("Dependencies versions")
    for package_name in DEPENDENCIES:
        print_package_version(package_name)
    print("=" * 120)
