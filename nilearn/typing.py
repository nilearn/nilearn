"""Types or Types alias used by Nilearn.


Many of those correspond to the types of variable
declared in nilearn._utils.doc.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Callable

from joblib.memory import Memory
from numpy import ndarray
from numpy.typing import DTypeLike

MemoryLevel = int
NJobs = int
Resume = bool
Standardize = bool
Verbose = int

# TODO update when dropping python 3.9
if sys.version_info[1] < 10:
    DataDir = (str, pathlib.Path)
    DType = (DTypeLike, str)
    HighPass = (float, int)
    HrfModel = (str, Callable, list)
    LowPass = (float, int)
    MemoryLike = (Memory, str, pathlib.Path)
    Resolution = int
    SmoothingFwhm = (float, int)
    Tr = (float, int)
    Url = str
    TargetAffine = ndarray
    TargetShape = (tuple, list)


else:
    from typing import TypeAlias

    DataDir: TypeAlias = str | pathlib.Path | None
    DType: TypeAlias = DTypeLike | str | None
    HrfModel = str | Callable | list | None
    HighPass: TypeAlias = float | int | None
    LowPass: TypeAlias = float | int | None
    MemoryLike: TypeAlias = Memory | str | pathlib.Path | None
    Resolution: TypeAlias = int | None
    SmoothingFwhm = float | int | None
    Url: TypeAlias = str | None
    TargetAffine: TypeAlias = ndarray | None
    TargetShape: TypeAlias = tuple | list | None
    Tr: TypeAlias = float | int | None
