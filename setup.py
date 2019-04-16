#! /usr/bin/env python

descr = """A set of python modules for neuroimaging..."""

import sys
import os

from setuptools import setup, find_packages


def load_version():
    """Executes nilearn/version.py in a globals dictionary and return it.

    Note: importing nilearn is not an option because there may be
    dependencies like nibabel which are not installed and
    setup.py is supposed to install them.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('nilearn', 'version.py')) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = 'nilearn'
DESCRIPTION = 'Statistical learning for neuroimaging in Python'
with open('README.rst') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Gael Varoquaux'
MAINTAINER_EMAIL = 'gael.varoquaux@normalesup.org'
URL = 'http://nilearn.github.io'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'http://nilearn.github.io'
VERSION = _VERSION_GLOBALS['__version__']


if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_nilearn_installing=True)

    install_requires = \
        ['%s>=%s' % (mod, meta['min_version'])
            for mod, meta in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']
            if not meta['required_at_installation']]

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
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
          ],
          packages=find_packages(),
          package_data={'nilearn.datasets.data': ['*.nii.gz', '*.csv'],
                        'nilearn.datasets.data.fsaverage5': ['*.gz'],
                        'nilearn.surface.data': ['*.csv'],
                        'nilearn.plotting.data.js': ['*.js'],
                        'nilearn.plotting.data.html': ['*.html'],
                        'nilearn.plotting.glass_brain_files': ['*.json'],
                        'nilearn.tests.data': ['*'],
                        'nilearn.image.tests.data': ['*.mgz'],
                        'nilearn.surface.tests.data': ['*.annot', '*.label'],
                        'nilearn.datasets.tests.data': ['*.*'],
                        'nilearn.datasets.description': ['*.rst']},
          install_requires=install_requires,)
