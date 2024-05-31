"""Path finding utilities."""

import glob
import os.path

from .helpers import stringify_path


def resolve_globbing(path):
    """Resolve globbing patterns in a path."""
    path = stringify_path(path)
    if isinstance(path, str):
        path_list = sorted(glob.glob(os.path.expanduser(path)))
        # Raise an error in case the list is empty.
        if len(path_list) == 0:
            raise ValueError(f"No files matching path: {path}")
        path = path_list

    return path
