from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import logistro

if TYPE_CHECKING:
    from typing import Any, Callable, MutableMapping, Sequence

_logger = logistro.getLogger(__name__)


class TmpDirWarning(UserWarning):
    """A warning if for whatever reason we can't eliminate the tmp dir."""


class TmpDirectory:
    """
    The python stdlib `TemporaryDirectory` wrapper for easier use.

    Python's `TemporaryDirectory` suffered a couple API changes that mean
    you can't call it the same way for similar versions. This wrapper is
    also much more aggressive about deleting the directory when it's done,
    not necessarily relying on OS functions.
    """

    temp_dir: tempfile.TemporaryDirectory[str]
    """A reference to the underlying python `TemporaryDirectory` implementation."""
    path: Path
    """The path to the temporary directory."""
    exists: bool
    """A flag to indicate if the directory still exists."""

    def __init__(self, path: str | None = None, *, sneak: bool = False):
        """
        Construct a wrapped `TemporaryDirectory`.

        Args:
            path: manually specify the directory to use
            sneak: (default False) avoid using /tmp
                Ubuntu's snap will sandbox /tmp

        """
        self._with_onexc = bool(sys.version_info[:3] >= (3, 12))
        args: MutableMapping[str, Any] = {}

        if path:
            args = {"dir": path}
        elif sneak:
            args = {"prefix": ".choreographer-", "dir": Path.home()}

        if platform.system() != "Windows":
            self.temp_dir = tempfile.TemporaryDirectory(**args)
        else:  # is windows
            vinfo = sys.version_info[:3]
            if vinfo >= (3, 12):
                self.temp_dir = tempfile.TemporaryDirectory(  # type: ignore [call-overload, unused-ignore]
                    delete=False,
                    ignore_cleanup_errors=True,
                    **args,
                )
            elif vinfo >= (3, 10):
                self.temp_dir = tempfile.TemporaryDirectory(  # type: ignore [call-overload, unused-ignore]
                    ignore_cleanup_errors=True,
                    **args,
                )
            else:
                self.temp_dir = tempfile.TemporaryDirectory(**args)

        self.path = Path(self.temp_dir.name)
        _logger.info(f"Temp directory created: {self.path}.")
        self.exists = True

    def _delete_manually(  # noqa: C901, PLR0912
        self,
        *,
        check_only: bool = False,
        quiet: bool = False,
    ) -> tuple[
        int,
        int,
        Sequence[tuple[Path, BaseException]],
    ]:
        if not self.path.exists():
            self.exists = False
            return 0, 0, []
        n_dirs = 0
        n_files = 0
        errors = []
        for root, dirs, files in os.walk(self.path, topdown=False):
            n_dirs += len(dirs)
            n_files += len(files)
            if not check_only:
                for f in files:
                    fp = Path(root) / f
                    _logger.debug2(f"Have file {fp}")
                    try:
                        fp.chmod(stat.S_IWUSR)
                        fp.unlink(missing_ok=True)
                        _logger.debug2("Deleted")
                    except Exception as e:  # noqa: BLE001 yes catch and report
                        errors.append((fp, e))
                for d in dirs:
                    fp = Path(root) / d
                    _logger.debug2(f"Have directory {fp}")
                    try:
                        fp.chmod(stat.S_IWUSR)
                        fp.rmdir()
                        _logger.debug2("Deleted")
                    except Exception as e:  # noqa: BLE001 yes catch and report
                        errors.append((fp, e))

            # clean up directory
        if not check_only:
            try:
                self.path.chmod(stat.S_IWUSR)
                self.path.rmdir()
            except Exception as e:  # noqa: BLE001 yes catch and report
                errors.append((self.path, e))

        if check_only:
            if n_dirs or n_files:
                self.exists = True
            else:
                self.exists = False
        elif errors:
            if not quiet:
                warnings.warn(  # noqa: B028
                    "The temporary directory could not be deleted, "
                    f"execution will continue. errors: {errors}",
                    TmpDirWarning,
                )
            self.exists = True
        else:
            self.exists = False

        return n_dirs, n_files, errors

    def clean(self) -> None:  # noqa: C901
        """Try several different ways to eliminate the temporary directory."""
        try:
            # no faith in this python implementation, always fails with windows
            # very unstable recently as well, lots new arguments in tempfile package
            if hasattr(self, "temp_dir") and self.temp_dir:
                self.temp_dir.cleanup()
            self.exists = False
            _logger.info("TemporaryDirectory.cleanup() worked.")
        except Exception as e:  # noqa: BLE001 we try many ways to clean, this is the first one
            _logger.info(f"TemporaryDirectory.cleanup() failed. Error {e}")

        # bad typing but tough
        def remove_readonly(
            func: Callable[[str], None],
            path: str | Path,
            _excinfo: Any,
        ) -> None:
            try:
                Path(path).chmod(stat.S_IWUSR)
                func(str(path))
            except FileNotFoundError:
                pass

        try:
            if self._with_onexc:
                shutil.rmtree(self.path, onexc=remove_readonly)  # type: ignore [call-arg, unused-ignore]
            else:
                shutil.rmtree(self.path, onerror=remove_readonly)
            self.exists = False
            if hasattr(self, "temp_dir"):
                del self.temp_dir
            _logger.info("shutil.rmtree worked.")
        except FileNotFoundError:
            self.exists = False
            if hasattr(self, "temp_dir"):
                del self.temp_dir
            _logger.info("shutil.rmtree worked.")
        except Exception as e:  # noqa: BLE001
            _logger.debug("Error during tmp file removal.", exc_info=e)
            self._delete_manually(check_only=True)
            if not self.exists:
                return
            _logger.info(f"shutil.rmtree() failed to delete temporary file. Error {e}")

            def extra_clean() -> None:
                i = 0
                tries = 4
                while self.path.exists() and i < tries:
                    _logger.info(f"Extra manual clean executing {i}.")
                    self._delete_manually(quiet=True)
                    i += 1
                    time.sleep(2)
                if self.path.exists():
                    self._delete_manually(quiet=False)

            # testing doesn't look threads so I guess we'll block
            extra_clean()
            if self.path.exists():
                _logger.warning("Temporary dictory couldn't be removed manually.")
