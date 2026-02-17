"""Adapter module for working with pydicom < 1.0 and >= 1.0

In what follows, "dicom is available" means we can import either a) ``dicom``
(pydicom < 1.0) or or b) ``pydicom`` (pydicom >= 1.0).

Regardless of whether dicom is available this module should be importable
without error, and always defines:

* have_dicom : True if we can import pydicom or dicom;
* pydicom : pydicom module or dicom module or None if not importable;
* read_file : ``read_file`` function if pydicom or dicom module is importable
  else None;
* tag_for_keyword : ``tag_for_keyword`` function if pydicom or dicom module
  is importable else None;

A test decorator is available in nibabel.nicom.tests:

* dicom_test : test decorator that skips test if dicom not available.

A deprecated copy is available here for backward compatibility.
"""

from __future__ import annotations

import warnings
from typing import Callable

from .deprecated import deprecate_with_version
from .optpkg import optional_package

warnings.warn(
    "We will remove the 'pydicom_compat' module from nibabel 7.0. "
    "Please consult pydicom's documentation for any future needs.",
    DeprecationWarning,
    stacklevel=2,
)

pydicom, have_dicom, _ = optional_package('pydicom')

read_file: Callable | None = None
tag_for_keyword: Callable | None = None
Sequence: type | None = None

if have_dicom:
    # Values not imported by default
    import pydicom.values  # type: ignore[import-not-found]
    from pydicom.dicomio import dcmread as read_file  # noqa:F401
    from pydicom.sequence import Sequence  # noqa:F401

    tag_for_keyword = pydicom.datadict.tag_for_keyword


@deprecate_with_version(
    'dicom_test has been moved to nibabel.nicom.tests', since='3.1', until='5.0'
)
def dicom_test(func):
    # Import locally to avoid circular dependency
    from .nicom.tests import dicom_test

    return dicom_test(func)
