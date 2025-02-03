"""Types or Types alias used by Nilearn.


Many of those correspond to the types of variable
declared in nilearn._utils.doc.
"""

from __future__ import annotations

import pathlib
from typing import TypeAlias

from joblib.memory import Memory

DataDir: TypeAlias = str | pathlib.Path | None
MemoryLike: TypeAlias = Memory | str | pathlib.Path | None
Resolution: TypeAlias = int | None
Resume: TypeAlias = bool
Url: TypeAlias = str | None
Verbose: TypeAlias = int
