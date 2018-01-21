# *- encoding: utf-8 -*-
"""
nilearn version, required package versions, and utilities for checking
"""
# Author: Loic Esteve, Ben Cipollini
# License: simplified BSD

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.4.0'

_NILEARN_INSTALL_MSG = 'See %s for installation information.' % (
    'http://nilearn.github.io/introduction.html#installation')

# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').
REQUIRED_MODULE_METADATA = (
    ('numpy', {
        'min_version': '1.6.1',
        'required_at_installation': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('scipy', {
        'min_version': '0.14',
        'required_at_installation': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('sklearn', {
        'min_version': '0.15',
        'required_at_installation': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('nibabel', {
        'min_version': '2.0.2',
        'required_at_installation': False}))

OPTIONAL_MATPLOTLIB_MIN_VERSION = '1.1.1'


def get_module_status(module_name):
    """
    Returns a dictionary 'module_status' containing a boolean status
    specifying whether module given in parameter module_name is installed
    or not. Also, returns import "module" if found otherwise empty.
    """
    module_status = {}
    try:
        module = __import__(module_name)
        module_status['installed'] = True
    except ImportError:
        module_status['installed'] = False
        module = ""
    return module, module_status


def _import_module_with_version_check(
        module_name,
        minimum_version,
        install_info=None):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    module, module_status = get_module_status(module_name)
    if module_status['installed'] is False:
        user_friendly_info = ('Module "{0}" could not be found. {1}').format(
            module_name,
            install_info or 'Please install it properly to use nilearn.')
        raise ImportError(user_friendly_info)

    if module_status['installed'] is True:
        # Avoid choking on modules with no __version__ attribute
        module_version = getattr(module, '__version__', '0.0.0')

    version_too_old = (not LooseVersion(module_version) >=
                       LooseVersion(minimum_version))

    if version_too_old:
        message = (
            'A {module_name} version of at least {minimum_version} '
            'is required to use nilearn. {module_version} was found. '
            'Please upgrade {module_name}').format(
                module_name=module_name,
                minimum_version=minimum_version,
                module_version=module_version)

        raise ImportError(message)

    return module


def _check_module_dependencies(is_nilearn_installing=False):
    """Throw an exception if nilearn dependencies are not installed.

    Parameters
    ----------
    is_nilearn_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.

    Throws
    -------
    ImportError
    """

    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if not (is_nilearn_installing and
                not module_metadata['required_at_installation']):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata['min_version'],
                install_info=module_metadata.get('install_info'))
