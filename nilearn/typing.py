"""Types or Types alias used by Nilearn.


Many of those correspond to the types of variable
declared in nilearn._utils.doc.
"""

from __future__ import annotations

import pathlib
import sys

from joblib.memory import Memory

MemoryLevel = int
Resume = bool
Verbose = int

# TODO update when dropping python 3.9
if sys.version_info[1] < 10:
    DataDir = (str, pathlib.Path)
    MemoryLike = (Memory, str, pathlib.Path)
    Resolution = int
    Url = str


else:
    from typing import TypeAlias

    DataDir: TypeAlias = str | pathlib.Path | None
    MemoryLike: TypeAlias = Memory | str | pathlib.Path | None
    Resolution: TypeAlias = int | None
    Url: TypeAlias = str | None

    # %(smoothing_fwhm)s

    # %(standardize)s

    # %(target_affine)s

    # %(target_shape)s

    # %(low_pass)s

    # %(high_pass)s

    # %(t_r)s

    # %(mask_strategy)s

    # %(memory)s

    # %(memory_level)s

    # %(n_jobs)s
    # %(smoothing_fwhm)s
    # %(standardize)s
    # %(target_affine)s
    # %(target_shape)s
    # %(low_pass)s
    # %(high_pass)s
    # %(t_r)s
    # %(mask_strategy)s
    # %(hrf_model)s
    # dtype
