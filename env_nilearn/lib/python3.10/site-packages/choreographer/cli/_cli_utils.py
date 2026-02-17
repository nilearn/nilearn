from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import sys
import urllib.request
import warnings
import zipfile
from functools import partial
from pathlib import Path

import logistro

from choreographer.cli.defaults import default_download_path

_logger = logistro.getLogger(__name__)

_chrome_for_testing_url = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"

supported_platform_strings = ["linux64", "win32", "win64", "mac-x64", "mac-arm64"]


def get_google_supported_platform_string() -> tuple[str, str, str, str]:
    arch_size_detected = "64" if sys.maxsize > 2**32 else "32"
    arch_detected = "arm" if platform.processor() == "arm" else "x"

    chrome_platform_detected: str | None = None
    if platform.system() == "Windows":
        chrome_platform_detected = "win" + arch_size_detected
    elif platform.system() == "Linux":
        chrome_platform_detected = "linux" + arch_size_detected
    elif platform.system() == "Darwin":
        chrome_platform_detected = "mac-" + arch_detected + arch_size_detected

    platform_string = ""
    if chrome_platform_detected in supported_platform_strings:
        platform_string = chrome_platform_detected

    return platform_string, arch_size_detected, platform.processor(), platform.system()


def get_chrome_download_path() -> Path | None:
    _chrome_platform_detected, _, _, _ = get_google_supported_platform_string()

    if not _chrome_platform_detected:
        return None

    _default_exe_path = Path()

    if platform.system().startswith("Linux"):
        _default_exe_path = (
            default_download_path / f"chrome-{_chrome_platform_detected}" / "chrome"
        )
    elif platform.system().startswith("Darwin"):
        _default_exe_path = (
            default_download_path
            / f"chrome-{_chrome_platform_detected}"
            / "Google Chrome for Testing.app"
            / "Contents"
            / "MacOS"
            / "Google Chrome for Testing"
        )
    elif platform.system().startswith("Win"):
        _default_exe_path = (
            default_download_path / f"chrome-{_chrome_platform_detected}" / "chrome.exe"
        )

    return _default_exe_path


# https://stackoverflow.com/questions/39296101/python-zipfile-removes-execute-permissions-from-binaries
class _ZipFilePermissions(zipfile.ZipFile):
    def _extract_member(self, member, targetpath, pwd):  # type: ignore [no-untyped-def]
        if not isinstance(member, zipfile.ZipInfo):
            member = self.getinfo(member)

        path = super()._extract_member(member, targetpath, pwd)  # type: ignore [misc]
        # High 16 bits are os specific (bottom is st_mode flag)
        attr = member.external_attr >> 16
        if attr != 0:
            Path(path).chmod(attr)
        return path


def get_chrome_sync(  # noqa: PLR0912, C901
    arch: str | None = None,
    i: int | None = None,
    path: str | Path = default_download_path,
    *,
    verbose: bool = False,
) -> Path | str:
    """Download chrome synchronously: see `get_chrome()`."""
    if not arch:
        arch, _, _, _ = get_google_supported_platform_string()

    if not arch:
        raise RuntimeError(
            "You must specify an arch, one of: "
            f"{', '.join(supported_platform_strings)}. "
            f"Detected {arch} is not supported.",
        )

    if isinstance(path, str):
        path = Path(path)
    if i:
        _logger.info("Loading chrome from list")
        browser_list = json.loads(
            urllib.request.urlopen(  # noqa: S310 audit url for schemes
                _chrome_for_testing_url,
            ).read(),
        )
        version_obj = browser_list["versions"][i]
    else:
        _logger.info("Using last known good version of chrome")
        with (
            Path(__file__).resolve().parent.parent
            / "resources"
            / "last_known_good_chrome.json"
        ).open() as f:
            version_obj = json.load(f)
    if verbose:
        print(version_obj["version"])  # noqa: T201 allow print in cli
        print(version_obj["revision"])  # noqa: T201 allow print in cli
    chromium_sources = version_obj["downloads"]["chrome"]
    url = ""
    for src in chromium_sources:
        if src["platform"] == arch:
            url = src["url"]
            break
    else:
        raise RuntimeError(
            "You must specify an arch, one of: "
            f"{', '.join(supported_platform_strings)}. "
            f"{arch} is not supported.",
        )

    if not path.exists():
        path.mkdir(parents=True)
    filename = path / "chrome.zip"
    with urllib.request.urlopen(url) as response, filename.open("wb") as out_file:  # noqa: S310 audit url
        shutil.copyfileobj(response, out_file)
    with _ZipFilePermissions(filename, "r") as zip_ref:
        zip_ref.extractall(path)
    filename.unlink()

    if arch.startswith("linux"):
        exe_name = path / f"chrome-{arch}" / "chrome"
    elif arch.startswith("mac"):
        exe_name = (
            path
            / f"chrome-{arch}"
            / "Google Chrome for Testing.app"
            / "Contents"
            / "MacOS"
            / "Google Chrome for Testing"
        )
    elif arch.startswith("win"):
        exe_name = path / f"chrome-{arch}" / "chrome.exe"
    else:
        raise RuntimeError("Couldn't calculate exe_name, unsupported architecture.")
    return exe_name


async def get_chrome(
    arch: str | None = None,
    i: int | None = None,
    path: str | Path = default_download_path,
    *,
    verbose: bool = False,
) -> Path | str:
    """
    Download google chrome from google-chrome-for-testing server.

    Args:
        arch: the target platform/os, as understood by google's json directory.
        i: the chrome version: -1 being the latest version, 0 being the oldest
           still in the testing directory.
        path: where to download it too (the folder).
        verbose: print out version found

    """
    loop = asyncio.get_running_loop()
    fn = partial(get_chrome_sync, arch=arch, i=i, path=path, verbose=verbose)
    return await loop.run_in_executor(
        executor=None,
        func=fn,
    )


def get_chrome_cli() -> None:
    if "ubuntu" in platform.version().lower():
        warnings.warn(  # noqa: B028
            "You are using `get_browser()` on Ubuntu."
            " Ubuntu is **very strict** about where binaries come from."
            " While sandbox is already off by default, do not set"
            " enable_sandbox to True OR you must install from Ubuntu's"
            " package manager.",
            UserWarning,
        )
    parser = argparse.ArgumentParser(
        description="Will download Chrome for testing. All arguments optional.",
        parents=[logistro.parser],
    )
    parser.add_argument(
        "--i",
        "-i",
        type=int,
        dest="i",
        help=(
            "Google offers thousands of chrome versions for download. "
            "'-i 0' is the oldest, '-i -1' is the newest: array syntax"
        ),
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        help="linux64|win32|win64|mac-x64|mac-arm64",
    )
    parser.add_argument(
        "--path",
        dest="path",
        help="Where to store the download.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Display found version number if using -i (to stdout)",
    )
    parser.set_defaults(path=default_download_path)
    parser.set_defaults(arch=None)
    parser.set_defaults(verbose=False)
    parsed = parser.parse_args()
    i = parsed.i
    arch = parsed.arch
    path = Path(parsed.path)
    verbose = parsed.verbose
    print(get_chrome_sync(arch=arch, i=i, path=path, verbose=verbose))  # noqa: T201 allow print in cli
