#! /usr/bin/env python

descr = """A set of python modules for neuroimaging..."""

import sys
import os
from setuptools import setup, find_packages

import nilearn

DISTNAME = 'nilearn'
DESCRIPTION = 'Statistical learning for neuroimaging in Python'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Gael Varoquaux'
MAINTAINER_EMAIL = 'gael.varoquaux@normalesup.org'
URL = 'http://nilearn.github.com'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'http://nilearn.github.com'
VERSION = nilearn.__version__

if __name__ == "__main__":
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(name=DISTNAME,
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
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
          ],
          packages=(find_packages() +
                    # The following are actually not packages
                    # and contain only data
                    ['nilearn/data',
                     'nilearn/plotting/glass_brain_files',
                     'nilearn/tests/data']),
          package_data={'nilearn/data': ['*.nii.gz'],
                        'nilearn/plotting/glass_brain_files': ['*.json'],
                        'nilearn/tests/data': ['*']},
    )
