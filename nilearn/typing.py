"""Types or Types alias used by Nilearn.


Many of those correspond to the types of variable
declared in nilearn._utils.doc.
"""

from __future__ import annotations

import pathlib

from joblib.memory import Memory

DataDir = str | pathlib.Path | None
MemoryLike = Memory | str | pathlib.Path | None
MemoryLevel = int
Resolution = int | None
Resume = bool
Url = str | None
Verbose = int
