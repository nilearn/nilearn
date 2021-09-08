""" Misc utilities for the library

Authors: load_confounds team
"""

def _flag_single_gifti(img_files):
    """Test if the paired input files are giftis."""
    flag_single_gifti = []  # gifti in pairs
    for img in img_files:
        ext = ".".join(img.split(".")[-2:])
        flag_single_gifti.append((ext == "func.gii"))
    return all(flag_single_gifti)
