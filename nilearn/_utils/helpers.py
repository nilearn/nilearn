import functools
import os
import sys
import warnings

from nilearn._utils.logger import find_stack_level
from nilearn._utils.versions import (
    OPTIONAL_MATPLOTLIB_MIN_VERSION,
    compare_version,
)


def set_mpl_backend(message=None) -> None:
    """Check if matplotlib is installed.

    If not installed, raise error and display warning to install necessary
    dependencies.

    If installed, check if the installed version complies with the minimum
    supported matplotlib version. If it does not, raise error; otherwise set
    the matplotlib backend.

    If current backend is not usable, switch to default "Agg" backend.

    Parameters
    ----------
    message: str, default=None
        Message to be prepended to standard warning when matplotlib is not
    installed.
    """
    # We are doing local imports here to avoid polluting our namespace
    try:
        import matplotlib
    except ImportError:
        warning = (
            "Some dependencies of nilearn.plotting package seem to be missing."
            "\nThey can be installed with:\n"
            " pip install 'nilearn[plotting]'"
        )
        if message is not None:
            warning = f"{message}\n{warning}"
        warnings.warn(warning, stacklevel=find_stack_level())
        raise
    else:
        # When matplotlib was successfully imported we need to check
        # that the version is greater that the minimum required one
        mpl_version = getattr(matplotlib, "__version__", "0.0.0")
        if not compare_version(
            mpl_version, ">=", OPTIONAL_MATPLOTLIB_MIN_VERSION
        ):
            raise ImportError(
                f"A matplotlib version of at least "
                f"{OPTIONAL_MATPLOTLIB_MIN_VERSION} "
                f"is required to use nilearn. {mpl_version} was found. "
                f"Please upgrade matplotlib."
            )
        current_backend = matplotlib.get_backend().lower()

        try:
            # Making sure the current backend is usable by matplotlib
            matplotlib.use(current_backend)
        except Exception:
            # If not, switching to default agg backend
            matplotlib.use("Agg")
        new_backend = matplotlib.get_backend().lower()

        if new_backend != current_backend:
            # Matplotlib backend has been changed, let's warn the user
            warnings.warn(
                f"Backend changed to {new_backend}...",
                stacklevel=find_stack_level(),
            )


def rename_parameters(
    replacement_params,
    end_version="future",
    lib_name="Nilearn",
):
    """Use this decorator to deprecate & replace specified parameters \
    in the decorated functions and methods without changing \
    function definition or signature.

    Parameters
    ----------
    replacement_params : Dict[string, string]
        Dict where the key-value pairs represent the old parameters
        and their corresponding new parameters.
        Example: {old_param1: new_param1, old_param2: new_param2,...}

    end_version : str {'future' | 'next' | <version>}, default='future'
        Version when using the deprecated parameters will raise an error.
        For informational purpose in the warning text.

    lib_name : str, default='Nilearn'
        Name of the library to which the decoratee belongs.
        For informational purpose in the warning text.

    """

    def _replace_params(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _warn_deprecated_params(
                replacement_params, end_version, lib_name, kwargs
            )
            kwargs = transfer_deprecated_param_vals(replacement_params, kwargs)
            return func(*args, **kwargs)

        return wrapper

    return _replace_params


def _warn_deprecated_params(
    replacement_params, end_version, lib_name, kwargs
) -> None:
    """Raise warnings about deprecated parameters, \
    for the decorator replace_parameters().

    Parameters
    ----------
    replacement_params : Dict[str, str]
        Dictionary of old_parameters as keys with replacement parameters
        as their corresponding values.

    end_version : str
        The version where use of the deprecated parameters will raise an error.
        For informational purpose in the warning text.

    lib_name : str
        Name of the library. For informational purpose in the warning text.

    kwargs : Dict[str, any]
        Dictionary of all the keyword args passed on the decorated function.

    """
    used_deprecated_params = set(kwargs).intersection(replacement_params)
    for deprecated_param_ in used_deprecated_params:
        replacement_param = replacement_params[deprecated_param_]
        param_deprecation_msg = (
            f'The parameter "{deprecated_param_}" '
            f"will be removed in {end_version} release of {lib_name}. "
            f'Please use the parameter "{replacement_param}" instead.'
        )
        warnings.warn(
            category=FutureWarning,
            message=param_deprecation_msg,
            stacklevel=find_stack_level(),
        )


def transfer_deprecated_param_vals(replacement_params, kwargs):
    """Reassigns new parameters \
    the values passed to their corresponding deprecated parameters \
    for the decorator replace_parameters().

    Parameters
    ----------
    replacement_params : Dict[str, str]
        Dictionary of old_parameters as keys with replacement parameters
        as their corresponding values.

    kwargs : Dict[str, any]
        Dictionary of all the keyword args passed on the decorated function.

    Returns
    -------
    kwargs : Dict[str, any]
        Dictionary of all the keyword args to be passed on
        to the decorated function, with old parameter names
        replaced by new parameters, with their values intact.

    """
    for old_param, new_param in replacement_params.items():
        old_param_val = kwargs.setdefault(old_param, None)
        if old_param_val is not None:
            kwargs[new_param] = old_param_val
        kwargs.pop(old_param)
    return kwargs


def remove_parameters(removed_params, reason, end_version="future"):
    """Use this decorator to deprecate \
    but not renamed parameters in the decorated functions and methods.

    Parameters
    ----------
    removed_params : list[string]
        List of old parameters to be removed.
        Example: [old_param1, old_param2, ...]

    reason : str
        Detailed reason of deprecated parameter and alternative solutions.

    end_version : str {'future' | 'next' | <version>}, default='future'
        Version when using the deprecated parameters will raise an error.
        For informational purpose in the warning text.

    """

    def _remove_params(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if found := set(removed_params).intersection(kwargs):
                message = (
                    f"Parameter(s) {', '.join(found)} "
                    f"will be removed in version {end_version}; "
                    f"{reason}"
                )
                warnings.warn(
                    category=FutureWarning,
                    message=message,
                    stacklevel=find_stack_level(),
                )
            return func(*args, **kwargs)

        return wrapper

    return _remove_params


def stringify_path(path):
    """Convert path-like objects to string.

    This is used to allow functions expecting string filesystem paths to accept
    objects using `__fspath__` protocol.

    Parameters
    ----------
    path : str or path-like object

    Returns
    -------
    str

    """
    return path.__fspath__() if isinstance(path, os.PathLike) else path


def is_matplotlib_installed():
    """Check if matplotlib is installed."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def check_matplotlib() -> None:
    """Check if matplotlib is installed, raise an error if not.

    Used in examples that require matplolib.
    """
    if not is_matplotlib_installed():
        raise RuntimeError(
            "This script needs the matplotlib library.\n"
            "You can install Nilearn "
            "and all its plotting dependencies with:\n"
            "pip install 'nilearn[plotting]'"
        )


def is_plotly_installed():
    """Check if plotly is installed."""
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        return False
    return True


def is_kaleido_installed():
    """Check if kaleido is installed."""
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False
    return True


def is_windows_platform():
    """Check if the current platform is Windows."""
    return os.name == "nt"


def is_gil_enabled():
    """Check if the Python GIL is enabled."""
    try:
        sys._is_gil_enabled()
    except AttributeError:
        # sys._is_gil_enabled does not exist in standard Python builds
        return True
