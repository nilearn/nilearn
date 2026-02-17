from __future__ import annotations

import os
import platform
from dataclasses import dataclass

chrome_names = (
    "chrome",
    "Chrome",
    "google-chrome",
    "google-chrome-stable",
    "Chrome.app",
    "Google Chrome",
    "Google Chrome.app",
    "Google Chrome for Testing",
)

chromium_names = ("chromium", "chromium-browser", "Chromium")

edge_names = (
    "msedge",
    "Microsoft Edge",
    "microsoft-edge",
    "microsoft-edge-beta",
    "microsoft-edge-dev",
)

brave_names = ("brave", "Brave Browser", "brave-browser")

vivaldi_names = ("vivaldi", "Vivaldi")


if platform.system() == "Windows":
    _windows_app_dirs = [
        _p
        for _p in [
            os.environ.get("PROGRAMFILES", ""),
            os.environ.get("PROGRAMFILES(X86)", ""),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        if _p
    ]
    _chrome_suffix = r"\Google\Chrome\Application\chrome.exe"
    typical_chrome_paths = tuple(_p + _chrome_suffix for _p in _windows_app_dirs)

    _chromium_suffix = r"\Chromium\Application\chrome.exe"
    typical_chromium_paths = tuple(_p + _chromium_suffix for _p in _windows_app_dirs)

    _edge_suffix = r"\Microsoft\Edge\Application\msedge.exe"
    typical_edge_paths = tuple(_p + _edge_suffix for _p in _windows_app_dirs)

    _brave_suffix = r"\BraveSoftware\Brave-Browser\Application\brave.exe"
    typical_brave_paths = tuple(_p + _edge_suffix for _p in _windows_app_dirs)

    _vivaldi_suffix = r"\Vivaldi\Application\vivaldi.exe"
    typical_vivaldi_paths = tuple(_p + _brave_suffix for _p in _windows_app_dirs)

elif platform.system() == "Linux":
    typical_chrome_paths = (
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chrome",
    )
    typical_chromium_paths = (
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    )
    typical_edge_paths = (
        "/usr/bin/microsoft-edge",
        "/usr/bin/microsoft-edge-beta",
        "/usr/bin/microsoft-edge-dev",
    )
    typical_brave_paths = (
        "/usr/bin/brave-browser",
        "/opt/brave.com/brave/brave-browser",
    )
    typical_vivaldi_paths = ("/usr/bin/vivaldi",)

else:  # assume mac, or system == "Darwin"
    typical_chrome_paths = (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    typical_chromium_paths = ("/Applications/Chromium.app/Contents/MacOS/Chromium",)

    typical_edge_paths = (
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    )

    typical_brave_paths = (
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    )
    typical_vivaldi_paths = ("/Applications/Vivaldi.app/Contents/MacOS/Vivaldi",)


@dataclass(frozen=True)
class BrowserInfo:
    __slots__ = ("exe_names", "ms_prog_id", "typical_paths")
    exe_names: tuple[str, ...]
    ms_prog_id: str
    typical_paths: tuple[str, ...]


chromium_based_browsers = {
    "chrome": BrowserInfo(chrome_names, "ChromeHTML", typical_chrome_paths),
    "chromium": BrowserInfo(chromium_names, "ChromiumHTML", typical_chromium_paths),
    "edge": BrowserInfo(edge_names, "MSEdgeHTM", typical_edge_paths),
    "brave": BrowserInfo(brave_names, "BraveHTML", typical_brave_paths),
    "vivaldi": BrowserInfo(vivaldi_names, "VivaldiHTM", typical_vivaldi_paths),
}
