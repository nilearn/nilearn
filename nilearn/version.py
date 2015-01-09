# *- encoding: utf-8 -*-
"""
nilearn version, required package versions, and utilities for checking
"""
# Author: Alexandre Abraham, Philippe Gervais, Ben Cipollini
# License: simplified BSD

__version__ = '0.1b1'


def _import_module_with_version_check(module_name, minimum_version):
    """Check that module is installed with a recent enough version
    """
    from distutils.version import LooseVersion

    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('{} could not be found, '
                              'please install it properly to use nilearn'
                              ).format(module_name)
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

def get_required_module_metadata():
    """Return all metadata needed for required modules:
        * minver: minimum required version of the module
        * manual: whether the module must be installed manually.

    See more info in check_module_dependencies docstring.
    """
    return {
        'numpy': {'minver': '1.6.0', 'manual_install': True},
        'scipy': {'minver': '0.9.0', 'manual_install': True},
        'sklearn': {'minver': '0.10', 'manual_install': True},
        'nibabel': {'minver': '1.1.0', 'manual_install': False},
        'gzip': {'minver': '0.0.0', 'manual_install': True}}

def check_module_dependencies(manual_install_only=False):
    """We want to check dependencies in the following scenarios:

        * When running installation, we want to:
            + Fail if manually-installed packages are not intalled
            + Communicate packages to install automatically

        * When running code from the nilearn package,
            + Fail if any needed module is missing
    """
    for module_name, module_metadata in get_required_module_metadata().iteritems():
        if module_metadata['manual_install'] and manual_install_only:
            # Skip check for manual install, when manual_install_only is specified.
            continue

        _import_module_with_version_check(module_name, module_metadata['minver'])

