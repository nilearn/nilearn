"""Types or Type aliases used by Nilearn.

Many of those correspond to the types of variable
declared in nilearn._utils.doc.

Several of them can be enforced at run time using
nilearn._utils.param_validaton.check_params.

To expand the functionality of check_params you need to:

-   describe the expected type for that parameter / attribute
    in this module ``nilearn.typing``
    It must be something that ``isinstance`` can handle.

-   expand the ``type_map`` dictionary of ``check_params``
    to pair the name of the parameter / attribute with its expected type.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Callable

import numpy as np
from joblib.memory import Memory
from numpy import ndarray
from numpy.typing import DTypeLike

BorderSize = int
Connected = int
Detrend = bool
LowerCutoff = float
MemoryLevel = int
NJobs = int
NPerm = int
Resume = bool
Standardize = bool
Tfce = bool
TwoSidedTest = bool
UpperCutoff = float
Verbose = int


# TODO update when dropping python 3.9
if sys.version_info[1] < 10:
    DataDir = (str, pathlib.Path)
    DType = DTypeLike
    HighPass = (float, int)
    HrfModel = (str, Callable, list)
    LowPass = (float, int)
    MemoryLike = (Memory, str, pathlib.Path)
    Opening = (bool, int)
    RandomState = (int, np.random.RandomState)
    Resolution = int
    SmoothingFwhm = (float, int)
    TargetAffine = ndarray
    TargetShape = (tuple, list)
    Threshold = (int, float, str)
    Title = str
    Tr = (float, int)
    Url = str
    Vmin = (float, int)
    Vmax = (float, int)


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
    RandomState = int | np.random.RandomState | None
    Opening: TypeAlias = bool | int
    Resolution: TypeAlias = int | None
    SmoothingFwhm: TypeAlias = float | int | None
    TargetAffine: TypeAlias = ndarray | None

    # Note that this is usable as for static type checking,
    # as type checkers will complain1
    # about using a generic and would prefer "list[int]" to "list".
    TargetShape: TypeAlias = tuple | list | None

    # str is too generic: should be Literal["auto"]
    Threshold: TypeAlias = int | float | str | None

    Title: TypeAlias = str | None
    Tr: TypeAlias = float | int | None
    Url: TypeAlias = str | None
    Vmin = float | int | None
    Vmax = float | int | None
