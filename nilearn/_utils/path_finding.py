"""Path finding utilities."""

import glob
from pathlib import Path

from .helpers import stringify_path


def resolve_globbing(path):
    """Resolve globbing patterns in a path."""
    path = stringify_path(path)
    if isinstance(path, str):
        expanded_path = Path(path).expanduser()
        path_list = sorted(glob.glob(str(expanded_path)))
        # Raise an error in case the list is empty.
        if len(path_list) == 0:
            raise ValueError(f"No files matching path: {path}")
        path_list = [Path(x) for x in path_list]
        path = path_list

    return path
