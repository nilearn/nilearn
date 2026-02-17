"""Module to help with deprecating objects and classes"""

from __future__ import annotations

import typing as ty
import warnings

from .deprecator import Deprecator
from .pkg_info import cmp_pkg_version

if ty.TYPE_CHECKING:
    P = ty.ParamSpec('P')


class ModuleProxy:
    """Proxy for module that may not yet have been imported

    Parameters
    ----------
    module_name : str
        Full module name e.g. ``nibabel.minc``

    Examples
    --------

    ::
        arr = np.arange(24).reshape((2, 3, 4))
        nifti1 = ModuleProxy('nibabel.nifti1')
        nifti1_image = nifti1.Nifti1Image(arr, np.eye(4))

    So, the ``nifti1`` object is a proxy that will import the required module
    when you do attribute access and return the attributes of the imported
    module.
    """

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name

    def __getattr__(self, key: str) -> ty.Any:
        mod = __import__(self._module_name, fromlist=[''])
        return getattr(mod, key)

    def __repr__(self) -> str:
        return f'<module proxy for {self._module_name}>'


class FutureWarningMixin:
    """Insert FutureWarning for object creation

    Examples
    --------
    >>> class C: pass
    >>> class D(FutureWarningMixin, C):
    ...     warn_message = "Please, don't use this class"

    Record the warning

    >>> with warnings.catch_warnings(record=True) as warns:
    ...     d = D()
    ...     warns[0].message.args[0]
    "Please, don't use this class"
    """

    warn_message = 'This class will be removed in future versions'

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        warnings.warn(self.warn_message, FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class VisibleDeprecationWarning(UserWarning):
    """Deprecation warning that will be shown by default

    Python >= 2.7 does not show standard DeprecationWarnings by default:

    http://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x

    Use this class for cases where we do want to show deprecations by default.
    """

    pass


deprecate_with_version = Deprecator(cmp_pkg_version)


def alert_future_error(
    msg: str,
    version: str,
    *,
    warning_class: type[Warning] = FutureWarning,
    error_class: type[Exception] = RuntimeError,
    warning_rec: str = '',
    error_rec: str = '',
    stacklevel: int = 2,
) -> None:
    """Warn or error with appropriate messages for changing functionality.

    Parameters
    ----------
    msg : str
        Description of the condition that led to the alert
    version : str
        NiBabel version at which the warning will become an error
    warning_class : subclass of Warning, optional
        Warning class to emit before version
    error_class : subclass of Exception, optional
        Error class to emit after version
    warning_rec : str, optional
        Guidance for suppressing the warning and avoiding the future error
    error_rec: str, optional
        Guidance for resolving the error
    stacklevel: int, optional
        Warnings stacklevel to provide; note that this will be incremented by
        1, so provide the stacklevel you would provide directly to warnings.warn()
    """
    if cmp_pkg_version(version) > 0:
        msg = f'{msg} This will error in NiBabel {version}. {warning_rec}'
        warnings.warn(msg.strip(), warning_class, stacklevel=stacklevel + 1)
    else:
        raise error_class(f'{msg} {error_rec}'.strip())
