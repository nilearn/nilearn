"""Utilities for nilearn users."""

from nilearn._utils.tags import InputTags, get_tag
from nilearn.utils.discovery import all_displays, all_estimators, all_functions

__all__ = [
    "InputTags",
    "all_displays",
    "all_estimators",
    "all_functions",
    "get_tag",
]
