"""
Path finding utilities
"""
import glob
import os.path


def _resolve_globbing(path):
    if isinstance(path, str):
        path_list = sorted(glob.glob(os.path.expanduser(path)))
        # Raise an error in case the list is empty.
        if len(path_list) == 0:
            raise ValueError("No files matching path: %s" % path)
        path = path_list

    return path
