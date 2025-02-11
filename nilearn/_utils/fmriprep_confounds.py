"""Misc utilities for function nilearn.interfaces.fmriprep.load_confounds.

Author: Hao-Ting Wang
"""


def flag_single_gifti(img_files):
    """Test if the paired input files are giftis."""
    # Possibly two gifti; if file is not correct, will be caught
    if isinstance(img_files[0], list):
        return False

    flag_single_gifti = []  # gifti in pairs
    for img in img_files:
        ext = ".".join(img.split(".")[-2:])
        flag_single_gifti.append(ext == "func.gii")
    return all(flag_single_gifti)


def is_camel_case(s):
    """Check if the given string is in camel case."""
    return s != s.lower() and s != s.upper() and "_" not in s


def to_camel_case(snake_str):
    """Convert camel to snake case."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components)
