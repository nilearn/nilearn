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
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

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
Url: TypeAlias = str | None
UpperCutoff: TypeAlias = float | np.floating
Verbose: TypeAlias = int | np.integer
Vmin = float | int | np.floating | np.integer | None
Vmax = float | int | np.floating | np.integer | None
