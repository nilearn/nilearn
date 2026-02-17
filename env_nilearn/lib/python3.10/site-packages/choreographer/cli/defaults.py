"""Defaults used when arguments not supplied."""

from pathlib import Path

default_download_path = Path(__file__).resolve().parent / "browser_exe"
"""The path where we download chrome if no path argument is supplied."""
