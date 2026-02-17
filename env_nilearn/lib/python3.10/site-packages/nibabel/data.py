# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities to find files from NIPY data packages"""

import configparser
import glob
import os
import sys
from os.path import join as pjoin

from packaging.version import Version

from .environment import get_nipy_system_dir, get_nipy_user_dir

DEFAULT_INSTALL_HINT = 'If you have the package, have you set the path to the package correctly?'


class DataError(Exception):
    pass


class BomberError(DataError, AttributeError):
    """Error when trying to access Bomber instance

    Should be instance of AttributeError to allow Python 3 inspect to do
    various ``hasattr`` checks without raising an error
    """

    pass


class Datasource:
    """Simple class to add base path to relative path"""

    def __init__(self, base_path):
        """Initialize datasource

        Parameters
        ----------
        base_path : str
           path to prepend to all relative paths

        Examples
        --------
        >>> from os.path import join as pjoin
        >>> repo = Datasource(pjoin('a', 'path'))
        >>> fname = repo.get_filename('somedir', 'afile.txt')
        >>> fname == pjoin('a', 'path', 'somedir', 'afile.txt')
        True
        """
        self.base_path = base_path

    def get_filename(self, *path_parts):
        """Prepend base path to `*path_parts`

        We make no check whether the returned path exists.

        Parameters
        ----------
        *path_parts : sequence of strings

        Returns
        -------
        fname : str
           result of ``os.path.join(*path_parts), with
           ``self.base_path`` prepended

        """
        return pjoin(self.base_path, *path_parts)

    def list_files(self, relative=True):
        """Recursively list the files in the data source directory.

        Parameters
        ----------
        relative: bool, optional
            If True, path returned are relative to the base path of
            the data source.

        Returns
        -------
        file_list: list of strings
            List of the paths of all the files in the data source.

        """
        out_list = list()
        for base, dirs, files in os.walk(self.base_path):
            if relative:
                base = base[len(self.base_path) + 1 :]
            out_list.extend(pjoin(base, filename) for filename in files)
        return out_list


class VersionedDatasource(Datasource):
    """Datasource with version information in config file"""

    def __init__(self, base_path, config_filename=None):
        """Initialize versioned datasource

        We assume that there is a configuration file with version
        information in datasource directory tree.

        The configuration file contains an entry like::

           [DEFAULT]
           version = 0.3

        The version should have at least a major and a minor version
        number in the form above.

        Parameters
        ----------
        base_path : str
           path to prepend to all relative paths
        config_filaname : None or str
           relative path to configuration file containing version

        """
        Datasource.__init__(self, base_path)
        if config_filename is None:
            config_filename = 'config.ini'
        self.config = configparser.ConfigParser()
        cfg_file = self.get_filename(config_filename)
        readfiles = self.config.read(cfg_file)
        if not readfiles:
            raise DataError(f'Could not read config file {cfg_file}')
        try:
            self.version = self.config.get('DEFAULT', 'version')
        except configparser.Error:
            raise DataError(f'Could not get version from {cfg_file}')
        version_parts = self.version.split('.')
        self.major_version = int(version_parts[0])
        self.minor_version = int(version_parts[1])
        self.version_no = float(f'{self.major_version}.{self.minor_version}')


def _cfg_value(fname, section='DATA', value='path'):
    """Utility function to fetch value from config file"""
    configp = configparser.ConfigParser()
    readfiles = configp.read(fname)
    if not readfiles:
        return ''
    try:
        return configp.get(section, value)
    except configparser.Error:
        return ''


def get_data_path():
    """Return specified or guessed locations of NIPY data files

    The algorithm is to return paths, extracted from strings, where
    strings are found in the following order:

    #. The contents of environment variable ``NIPY_DATA_PATH``
    #. Any section = ``DATA``, key = ``path`` value in a ``config.ini``
       file in your nipy user directory (found with
       ``get_nipy_user_dir()``)
    #. Any section = ``DATA``, key = ``path`` value in any files found
       with a ``sorted(glob.glob(os.path.join(sys_dir, '*.ini')))``
       search, where ``sys_dir`` is found with ``get_nipy_system_dir()``
    #. If ``sys.prefix`` is ``/usr``, we add
       ``/usr/local/share/nipy``. We need this because Python 2.6 in
       Debian / Ubuntu does default installs to ``/usr/local``.
    #. The result of ``get_nipy_user_dir()``

    Therefore, any paths found in ``NIPY_DATA_PATH`` will be searched
    before paths found in the user directory ``config.ini``

    Parameters
    ----------
    None

    Returns
    -------
    paths : sequence of paths

    Examples
    --------
    >>> pth = get_data_path()

    Notes
    -----
    We have to add ``/usr/local/share/nipy`` if sys.prefix is ``/usr``,
    because Debian has patched distutils in Python 2.6 to do default
    distutils installs there:

    * https://www.debian.org/doc/packaging-manuals/python-policy/ap-packaging_tools.html#s-distutils
    * https://www.mail-archive.com/debian-python@lists.debian.org/msg05084.html
    """
    paths = []
    try:
        var = os.environ['NIPY_DATA_PATH']
    except KeyError:
        pass
    else:
        if var:
            paths = var.split(os.path.pathsep)
    np_cfg = pjoin(get_nipy_user_dir(), 'config.ini')
    np_etc = get_nipy_system_dir()
    config_files = sorted(glob.glob(pjoin(np_etc, '*.ini')))
    for fname in [np_cfg] + config_files:
        var = _cfg_value(fname)
        if var:
            paths += var.split(os.path.pathsep)
    paths.append(pjoin(sys.prefix, 'share', 'nipy'))
    if sys.prefix == '/usr':
        paths.append(pjoin('/usr/local', 'share', 'nipy'))
    paths.append(pjoin(get_nipy_user_dir()))
    return paths


def find_data_dir(root_dirs, *names):
    """Find relative path given path prefixes to search

    We raise a DataError if we can't find the relative path

    Parameters
    ----------
    root_dirs : sequence of strings
       sequence of paths in which to search for data directory
    *names : sequence of strings
       sequence of strings naming directory to find. The name to search
       for is given by ``os.path.join(*names)``

    Returns
    -------
    data_dir : str
       full path (root path added to `*names` above)

    """
    ds_relative = pjoin(*names)
    for path in root_dirs:
        pth = pjoin(path, ds_relative)
        if os.path.isdir(pth):
            return pth
    raise DataError(
        f'Could not find datasource "{ds_relative}" in '
        f'data path "{os.path.pathsep.join(root_dirs)}"'
    )


def make_datasource(pkg_def, **kwargs):
    """Return datasource defined by `pkg_def` as found in `data_path`

    `data_path` is the only allowed keyword argument.

    `pkg_def` is a dictionary with at least one key - 'relpath'.  'relpath' is
    a relative path with unix forward slash separators.

    The relative path to the data is found with::

        names = pkg_def['name'].split('/')
        rel_path = os.path.join(names)

    We search for this relative path in the list of paths given by `data_path`.
    By default `data_path` is given by ``get_data_path()`` in this module.

    If we can't find the relative path, raise a DataError

    Parameters
    ----------
    pkg_def : dict
       dict containing at least the key 'relpath'. 'relpath' is the data path
       of the package relative to `data_path`.  It is in unix path format
       (using forward slashes as directory separators).  `pkg_def` can also
       contain optional keys 'name' (the name of the package), and / or a key
       'install hint' that we use in the returned error message from trying to
       use the resulting datasource
    data_path : sequence of strings or None, optional
       sequence of paths in which to search for data.  If None (the
       default), then use ``get_data_path()``

    Returns
    -------
    datasource : ``VersionedDatasource``
       An initialized ``VersionedDatasource`` instance
    """
    if any(key for key in kwargs if key != 'data_path'):
        raise ValueError('Unexpected keyword argument(s)')
    data_path = kwargs.get('data_path')
    if data_path is None:
        data_path = get_data_path()
    unix_relpath = pkg_def['relpath']
    names = unix_relpath.split('/')
    try:
        pth = find_data_dir(data_path, *names)
    except DataError as e:
        pth = [pjoin(this_data_path, *names) for this_data_path in data_path]
        pkg_hint = pkg_def.get('install hint', DEFAULT_INSTALL_HINT)
        msg = f'{e}; Is it possible you have not installed a data package?'
        if 'name' in pkg_def:
            msg += f"\n\nYou may need the package \"{pkg_def['name']}\""
        if pkg_hint is not None:
            msg += f'\n\n{pkg_hint}'
        raise DataError(msg)
    return VersionedDatasource(pth)


class Bomber:
    """Class to raise an informative error when used"""

    def __init__(self, name, msg):
        self.name = name
        self.msg = msg

    def __getattr__(self, attr_name):
        """Raise informative error accessing not-found attributes"""
        raise BomberError(
            f'Trying to access attribute "{attr_name}" of '
            f'non-existent data "{self.name}"\n\n{self.msg}\n'
        )


def datasource_or_bomber(pkg_def, **options):
    """Return a viable datasource or a Bomber

    This is to allow module level creation of datasource objects.  We
    create the objects, so that, if the data exist, and are the correct
    version, the objects are valid datasources, otherwise, they
    raise an error on access, warning about the lack of data or the
    version numbers.

    The parameters are as for ``make_datasource`` in this module.

    Parameters
    ----------
    pkg_def : dict
       dict containing at least key 'relpath'. Can optionally have keys 'name'
       (package name),  'install hint' (for helpful error messages) and 'min
       version' giving the minimum necessary version string for the package.
    data_path : sequence of strings or None, optional

    Returns
    -------
    ds : datasource or ``Bomber`` instance
    """
    unix_relpath = pkg_def['relpath']
    version = pkg_def.get('min version')
    pkg_hint = pkg_def.get('install hint', DEFAULT_INSTALL_HINT)
    names = unix_relpath.split('/')
    sys_relpath = os.path.sep.join(names)
    try:
        ds = make_datasource(pkg_def, **options)
    except DataError as e:
        return Bomber(sys_relpath, str(e))
    # check version
    if version is None or Version(ds.version) >= Version(version):
        return ds
    if 'name' in pkg_def:
        pkg_name = pkg_def['name']
    else:
        pkg_name = 'data at ' + unix_relpath
    msg = f'{pkg_name} is version {ds.version} but we need version >= {version}\n\n{pkg_hint}'
    return Bomber(sys_relpath, DataError(msg))
