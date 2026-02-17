from __future__ import annotations

import os
import platform
import re
import shutil
from typing import TYPE_CHECKING

import logistro

from choreographer.cli._cli_utils import get_chrome_download_path

_logger = logistro.getLogger()

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Sequence


def _is_exe(path: str | Path) -> bool:
    try:
        return os.access(path, os.X_OK)
    except:  # noqa: E722 bare except ok, weird errors, best effort.
        return False


def _which_from_windows_reg(key: str) -> str | None:
    try:
        import winreg  # noqa: PLC0415 don't import if not windows pls

        command = winreg.QueryValueEx(  # type: ignore [attr-defined]
            winreg.OpenKey(  # type: ignore [attr-defined]
                winreg.HKEY_CLASSES_ROOT,  # type: ignore [attr-defined]
                f"{key}\\shell\\open\\command",
                0,
                winreg.KEY_READ,  # type: ignore [attr-defined]
            ),
            "",
        )[0]
        exe = re.search('"(.*?)"', command).group(1)  # type: ignore [union-attr]
    except Exception:  # noqa: BLE001 don't care why, best effort search
        return None

    return exe


def browser_which(
    executable_names: Sequence[str],
    *,
    skip_local: bool = False,
    ms_prog_id: str | None = None,
    verify_local: bool = False,
) -> str | None:
    """
    Look for and return first name found in PATH.

    Args:
        executable_names: the list of names to look for
        skip_local: (default False) don't look for a choreo download of anything.
        ms_prog_id: A windows registry ID string to lookup program paths
        verify_local: (default False) force using local install.

    """
    _logger.debug(f"Looking for browser, skipping local? {skip_local}")
    path = None

    if isinstance(executable_names, str):
        executable_names = [executable_names]

    if skip_local:
        _logger.debug("Skipping searching for local download of chrome.")
        if verify_local:
            raise ValueError("Cannot set both skip_local and verify_local.")
    else:
        local_chrome = get_chrome_download_path()
        _logger.debug(f"Looking for at local chrome download path: {local_chrome}")
        if local_chrome is not None and local_chrome.exists():
            _logger.debug("Returning local chrome.")
            return str(local_chrome)
        else:
            _logger.debug(f"Local chrome not found at path: {local_chrome}.")
            if verify_local:
                raise RuntimeError("verify_local set to True, local not found.")

    if platform.system() == "Windows":
        os.environ["NoDefaultCurrentDirectoryInExePath"] = "0"  # noqa: SIM112 var name set by windows
        if (
            ms_prog_id
            and (path := _which_from_windows_reg(ms_prog_id))
            and _is_exe(path)
        ):
            return path

    for exe in executable_names:
        path = shutil.which(exe)
        if path and _is_exe(path):
            return path

    return None


def get_browser_path(*args: Any, **kwargs: Any) -> str | None:  # noqa: D417, don't pass args explicitly
    """
    Call `browser_which()` but check for user override first.

    It looks for the browser in path.

    Accepts the same arguments as `browser_which':

    Args:
        executable_names: the list of names to look for
        skip_local: (default False) don't look for a choreo download of anything.

    """
    return os.environ.get("BROWSER_PATH", browser_which(*args, **kwargs))
