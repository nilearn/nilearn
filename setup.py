#! /usr/bin/env python

# noqa: D100

import os

from setuptools import find_packages, setup


def load_version():
    """Execute nilearn/version.py in a globals dictionary and return it.

    Note: importing nilearn is not an option because there may be
    dependencies like nibabel which are not installed and
    setup.py is supposed to install them.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join("nilearn", "version.py")) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = "nilearn"
DESCRIPTION = "Statistical learning for neuroimaging in Python"
with open("README.rst") as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = "Bertrand Thirion"
MAINTAINER_EMAIL = "bertrand.thirion@inria.fr"
URL = "https://nilearn.github.io"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://nilearn.github.io"
VERSION = _VERSION_GLOBALS["__version__"]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        packages=find_packages(),
        package_data={
            "nilearn.datasets.data": ["*.nii.gz", "*.csv", "*.txt"],
            "nilearn.datasets.data.fsaverage5": ["*.gz"],
            "nilearn.surface.data": ["*.csv"],
            "nilearn.plotting.data.js": ["*.js"],
            "nilearn.plotting.data.html": ["*.html"],
            "nilearn.plotting.glass_brain_files": ["*.json"],
            "nilearn.tests.data": ["*"],
            "nilearn.image.tests.data": ["*.mgz"],
            "nilearn.surface.tests.data": ["*.annot", "*.label"],
            "nilearn.datasets.tests.data": ["*.*"],
            "nilearn.datasets.tests.data.archive_contents": ["*"],
            "nilearn.datasets.tests.data.archive_contents.nyu_rest": ["*"],
            "nilearn.datasets.tests.data.archive_contents.test_examples": [
                "*"
            ],
            "nilearn.datasets.description": ["*.rst"],
            "nilearn.reporting.data.html": ["*.html"],
            "nilearn.glm.tests": ["*.nii.gz", "*.npz"],
            "nilearn.reporting.glm_reporter_templates": ["*.html"],
        },
    )
