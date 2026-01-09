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

import pathlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from joblib.memory import Memory
from nibabel import Nifti1Image
from numpy import ndarray
from numpy.typing import DTypeLike

Annotate: TypeAlias = bool
BgOnData: TypeAlias = bool
BorderSize: TypeAlias = int | np.integer
ColorBar: TypeAlias = bool
ClusterThreshold: TypeAlias = int | np.integer
Connected: TypeAlias = bool
CopyHeader: TypeAlias = bool
DType: TypeAlias = DTypeLike | None
DataDir: TypeAlias = str | pathlib.Path | None
Detrend: TypeAlias = bool
DrawCross: TypeAlias = bool
ForceResample: TypeAlias = bool
HeightControl = Literal[None, "fpr", "fdr", "bonferroni"]
# Note that for HrfModel
# str is too generic here
# and it should actually be Literal["spm", "glover", ...]
# if we wanted to use proper type annotation
HrfModel: TypeAlias = str | Callable | list | None

HighPass: TypeAlias = float | int | np.floating | np.integer | None
KeepMaskedLabels: TypeAlias = bool
KeepMaskedMaps: TypeAlias = bool
LowerCutoff: TypeAlias = float | np.floating
LowPass: TypeAlias = float | int | np.floating | np.integer | None
MemoryLike: TypeAlias = Memory | str | pathlib.Path | None
MemoryLevel: TypeAlias = int | np.integer
NJobs: TypeAlias = int | np.integer
NPerm: TypeAlias = int | np.integer
NiimgLike = (Nifti1Image, str, Path)
Opening: TypeAlias = bool | int | np.integer
Radiological: TypeAlias = bool
RandomState: TypeAlias = (
    int | np.floating | np.integer | np.random.RandomState | None
)
Resolution: TypeAlias = int | np.integer | None
Resume: TypeAlias = bool
ScreeningPercentile: TypeAlias = float | int | np.floating | np.integer | None
SmoothingFwhm: TypeAlias = float | int | np.floating | np.integer | None
Standardize: TypeAlias = Literal[
    "zscore", "zscore_sample", "psc", True, False, None
]
StandardizeConfounds: TypeAlias = bool
TargetAffine: TypeAlias = ndarray | list | tuple | None

# Note that this is usable as for static type checking,
# as type checkers will complain
# about using a generic and would prefer "list[int]" to "list".
TargetShape: TypeAlias = tuple | list | ndarray | None

Threshold: TypeAlias = float | int | np.floating | np.integer | str | None

Tfce: TypeAlias = bool
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
TwoSidedTest: TypeAlias = bool
Url: TypeAlias = str | None
UpperCutoff: TypeAlias = float | np.floating
Verbose: TypeAlias = bool | int | np.integer
Vmin: TypeAlias = float | int | np.floating | np.integer | None
Vmax: TypeAlias = float | int | np.floating | np.integer | None
