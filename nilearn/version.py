# *- encoding: utf-8 -*-
"""
nilearn version, required package versions, and utilities for checking
"""
# Author: Alexandre Abraham, Philippe Gervais, Ben Cipollini
# License: simplified BSD

__version__ = '0.1b1'

# All metadata needed for required modules:
#    * minver: minimum required version of the module
#    * manual: whether the module must be installed manually
# See more info in check_module_dependencies docstring.
#
# This is a list to preserve order, and to avoid the Python 2.7
#   dependency of collections.OrderedDict
_NILEARN_INSTALL_MSG = 'See %s for installation information.' % (
    'http://nilearn.github.io/introduction.html#installation')
REQUIRED_MODULE_METADATA = (
    ('numpy', {
        'minver': '1.6.0',
        'manual_install': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('scipy', {
        'minver': '0.9.0',
        'manual_install': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('sklearn', {
        'minver': '0.10',
        'manual_install': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('nibabel', {
        'minver': '1.1.0',
        'manual_install': False}),
    ('gzip', {
        'minver': '0.0.0',
        'manual_install': True,
        'install_info': 'Use a python version compiled with gzip.'}))


def _import_module_with_version_check(
        module_name,
        minimum_version,
        install_info=None):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('Module "{}" could not be found.  {}').format(
            module_name,
            install_info or 'Please install it properly to use nilearn.')
        exc.args += (user_friendly_info,)
        raise

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


def _check_module_dependencies(preinstalled_modules_only=False):
    """We want to check dependencies in the following scenarios:

        * When running installation, we want to:
            + Fail if manually-installed packages are not installed
            + Communicate packages to install automatically

        * When running code from the nilearn package,
            + Fail if any needed module is missing
    """
    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if preinstalled_modules_only and not module_metadata['manual_install']:
            # Skip check for auto-installed modules,
            #   when preinstalled_modules_only is specified.
            continue

        _import_module_with_version_check(
            module_name=module_name,
            minimum_version=module_metadata['minver'],
            install_info=module_metadata.get('install_info'))
