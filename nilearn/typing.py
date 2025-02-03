"""Types or Types alias used by Nilearn.


Many of those correspond to the types of variable
declared in nilearn._utils.doc.
"""

from __future__ import annotations

import pathlib
import sys

from joblib.memory import Memory
from numpy import ndarray
from numpy.typing import DTypeLike

HrfModel = str
MemoryLevel = int
NJobs = int
Resume = bool
Standardize = bool
Verbose = int

# TODO update when dropping python 3.9
if sys.version_info[1] < 10:
    DataDir = (str, pathlib.Path)
    DType = (DTypeLike, str)
    HighPass = float
    LowPass = float
    MemoryLike = (Memory, str, pathlib.Path)
    Resolution = int
    SmoothingFwhm = float
    Tr = float
    Url = str
    TargetAffine = ndarray
    TargetShape = (tuple, list)


else:
    from typing import TypeAlias

    DataDir: TypeAlias = str | pathlib.Path | None
    DType: TypeAlias = DTypeLike | str | None
    HighPass: TypeAlias = float | None
    LowPass: TypeAlias = float | None
    MemoryLike: TypeAlias = Memory | str | pathlib.Path | None
    Resolution: TypeAlias = int | None
    SmoothingFwhm = float
    Url: TypeAlias = str | None
    TargetAffine: TypeAlias = ndarray | None
    TargetShape: TypeAlias = tuple[int, int, int] | list[int] | None
    Tr: TypeAlias = float | None
