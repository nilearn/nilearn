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
from pathlib import Path
from typing import Callable

import numpy as np
from joblib.memory import Memory
from nibabel import Nifti1Image
from numpy import ndarray
from numpy.typing import DTypeLike

Annotate = bool
BgOnData = bool
ColorBar = bool
Connected = bool
Detrend = bool
DrawCross = bool
KeepMaskedLabels = bool
KeepMaskedMaps = bool
NiimgLike = (Nifti1Image, str, Path)
Radiological = bool
Resume = bool
Standardize = bool
StandardizeConfounds = bool
Tfce = bool
TwoSidedTest = bool


# TODO (python >= 3.10) update when dropping python 3.9
if sys.version_info[1] < 10:
    BorderSize = (int, np.integer)
    DataDir = (str, pathlib.Path)
    DType = DTypeLike
    HighPass = (float, int, np.floating, np.integer)
    HrfModel = (str, Callable, list)
    LowPass = (float, int, np.floating, np.integer)
    LowerCutoff = (float, np.floating)
    MemoryLevel = (int, np.integer)
    MemoryLike = (Memory, str, pathlib.Path)
    NJobs = (int, np.integer)
    NPerm = (int, np.integer)
    Opening = (bool, int, np.integer)
    RandomState = (int, np.integer, np.random.RandomState)
    Resolution = (int, np.integer)
    SmoothingFwhm = (float, int, np.floating, np.integer)
    ScreeningPercentile = (float, int, np.floating, np.integer)
    TargetAffine = ndarray
    TargetShape = (tuple, list)
    Threshold = (float, int, str, np.floating, np.integer)
    Title = str
    Tr = (float, int, np.floating, np.integer)
    Transparency = (
        float,
        int,
        np.floating,
        np.integer,
        str,
        Path,
        Nifti1Image,
        Path,
    )
    TransparencyRange = (list, tuple)
    Url = str
    UpperCutoff = (float, np.floating)
    Verbose = (int, np.integer)
    Vmin = (float, int, np.floating, np.integer)
    Vmax = (float, int, np.floating, np.integer)


else:
    from typing import TypeAlias

    BorderSize: TypeAlias = int | np.integer
    DataDir: TypeAlias = str | pathlib.Path | None
    DType: TypeAlias = DTypeLike | None

    # Note that for HrfModel
    # str is too generic here
    # and it should actually be Literal["spm", "glover", ...]
    # if we wanted to use proper type annotation
    HrfModel: TypeAlias = str | Callable | list | None

    HighPass: TypeAlias = float | int | np.floating | np.integer | None
    LowerCutoff: TypeAlias = float | np.floating
    LowPass: TypeAlias = float | int | np.floating | np.integer | None
    MemoryLike: TypeAlias = Memory | str | pathlib.Path | None
    MemoryLevel: TypeAlias = int | np.integer
    NJobs: TypeAlias = int | np.integer
    NPerm: TypeAlias = int | np.integer
    RandomState = int | np.floating | np.integer | np.random.RandomState | None
    Opening: TypeAlias = bool | int | np.integer
    Resolution: TypeAlias = int | np.integer | None
    ScreeningPercentile: TypeAlias = (
        float | int | np.floating | np.integer | None
    )
    SmoothingFwhm: TypeAlias = float | int | np.floating | np.integer | None
    TargetAffine: TypeAlias = ndarray | None

    # Note that this is usable as for static type checking,
    # as type checkers will complain1
    # about using a generic and would prefer "list[int]" to "list".
    TargetShape: TypeAlias = tuple | list | None

    # str is too generic: should be Literal["auto"]
    Threshold: TypeAlias = float | int | np.floating | np.integer | str | None

    Title: TypeAlias = str | None
    Tr: TypeAlias = float | int | np.floating | np.integer | None
    Transparency: TypeAlias = (
        float
        | int
        | np.floating
        | np.integer
        | str
        | Path
        | Nifti1Image
        | Path
        | None
    )
    TransparencyRange: TypeAlias = list | tuple | None
    Url: TypeAlias = str | None
    UpperCutoff: TypeAlias = float | np.floating
    Verbose: TypeAlias = int | np.integer
    Vmin = float | int | np.floating | np.integer | None
    Vmax = float | int | np.floating | np.integer | None
