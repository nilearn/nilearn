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
    DType = DTypeLike
    HighPass = (float, int)
    HrfModel = (str, Callable, list)
    LowPass = (float, int)
    MemoryLike = (Memory, str, pathlib.Path)
    Resolution = int
    SmoothingFwhm = (float, int)
    TargetAffine = ndarray
    TargetShape = (tuple, list)
    Tr = (float, int)
    Url = str


else:
    from typing import TypeAlias

    DataDir: TypeAlias = str | pathlib.Path | None
    DType: TypeAlias = DTypeLike | None

    # Note that for HrfModel
    # str is too generic here
    # and it should actually be Literal["spm", "glover", ...]
    # if we wanted to use proper type annotation
    HrfModel: TypeAlias = str | Callable | list | None

    HighPass: TypeAlias = float | int | None
    LowPass: TypeAlias = float | int | None
    MemoryLike: TypeAlias = Memory | str | pathlib.Path | None
    Resolution: TypeAlias = int | None
    SmoothingFwhm: TypeAlias = float | int | None
    TargetAffine: TypeAlias = ndarray | None

    # Note that this is usable as for static type checking,
    # as type checkers will complain1
    # about using a generic and would prefer "list[int]" to "list".
    TargetShape: TypeAlias = tuple | list | None

    Tr: TypeAlias = float | int | None
    Url: TypeAlias = str | None
