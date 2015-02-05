# *- encoding: utf-8 -*-
"""
nilearn version, required package versions, and utilities for checking
"""
# Author: Loïc Estève, Ben Cipollini
# License: simplified BSD

__version__ = '0.1b1'

_NILEARN_INSTALL_MSG = 'See %s for installation information.' % (
    'http://nilearn.github.io/introduction.html#installation')

# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').  We avoid using
#   collections.OrderedDict to preserve Python 2.6 compatibility.
REQUIRED_MODULE_METADATA = (
    ('numpy', {
        'min_version': '1.6.0',
        'required_at_installation': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('scipy', {
        'min_version': '0.11.0',
        'required_at_installation': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('sklearn', {
        'min_version': '0.12.1',
        'required_at_installation': True,
        'install_info': _NILEARN_INSTALL_MSG}),
    ('nibabel', {
        'min_version': '1.1.0',
        'required_at_installation': False}),)


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
        user_friendly_info = ('Module "{0}" could not be found. {1}').format(
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
