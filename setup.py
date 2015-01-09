#! /usr/bin/env python

descr = """A set of python modules for neuroimaging..."""

import sys
import os

from setuptools import setup, find_packages

# Make the current nilearn sources importable.
__curdir__ = os.path.dirname(__file__)
sys.path.insert(1, __curdir__)

from nilearn import version

def get_version():
    """Returns the version found in nilearn/version.py

    Importing nilearn is not an option because there may dependencies
    like nibabel which are not installed and setup.py is supposed to
    install them.
    """
    locals_dict = {}
    globals_dict = {}
    with open(os.path.join(__curdir__, 'nilearn', 'version.py')) as fp:
        exec(fp.read(), globals_dict, locals_dict)

    return locals_dict['__version__']

DISTNAME = 'nilearn'
DESCRIPTION = 'Statistical learning for neuroimaging in Python'
LONG_DESCRIPTION = open(os.path.join(__curdir__, 'README.rst')).read()
MAINTAINER = 'Gael Varoquaux'
MAINTAINER_EMAIL = 'gael.varoquaux@normalesup.org'
URL = 'http://nilearn.github.com'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'http://nilearn.github.com'
VERSION = get_version()

if __name__ == "__main__":
    version.check_module_dependencies(manual_install_only=True)

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    # Convert module metadata into list of {mod_name}>={minver}
    #   as required by install_requires
    formatted_dependencies = ['%s>=%s' % (mod, meta['minver'])
        for mod, meta in version.get_required_module_metadata().iteritems()
        if not meta['manual_install']]

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
          packages=find_packages(),
          package_data={'nilearn.data': ['*.nii.gz'],
                        'nilearn.plotting.glass_brain_files': ['*.json'],
                        'nilearn.tests.data': ['*']},
          install_requires=formatted_dependencies,
    )
