"""Path finding utilities."""

import glob
import os
from pathlib import Path
from typing import TypeVar, Union, overload

from nilearn._utils.helpers import stringify_path

T = TypeVar("T")  # Generic type variable for non-path-like inputs


@overload
def resolve_globbing(
    path: str,
) -> list[os.PathLike[str]]: ...


@overload
def resolve_globbing(
    path: T,
) -> T: ...


def resolve_globbing(path: Union[T, str, os.PathLike[str]]):
    """Resolve globbing patterns in a path."""
    stringified = stringify_path(path)

    if not isinstance(stringified, str):
        return stringified

    expanded_path = Path(stringified).expanduser()

    str_list = sorted(glob.glob(str(expanded_path)))

    # Raise an error in case the list is empty.
    if len(str_list) == 0:
        raise ValueError(f"No files matching path: {stringified}")

    path_list = [Path(x) for x in str_list]

    return path_list
