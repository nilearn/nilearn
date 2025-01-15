"""Downloading NeuroImaging datasets: utility functions."""

import os
from pathlib import Path
from warnings import warn

from .._utils import fill_doc

_GENERAL_MESSAGE = (
    "The import path 'nilearn.datasets.utils'\n"
    "will be deprecated in version 0.13.\n"
    "Importing from 'nilearn.datasets.utils will be possible\n"
    "at least until release 0.13.0."
)


@fill_doc
def get_data_dirs(data_dir=None):
    """Return the directories in which nilearn looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    Parameters
    ----------
    %(data_dir)s

    Returns
    -------
    paths : list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :

    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NILEARN_SHARED_DATA
    4. the user environment variable NILEARN_DATA
    5. nilearn_data in the user home folder

    """
    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(str(data_dir).split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv("NILEARN_SHARED_DATA")
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv("NILEARN_DATA")
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        paths.append(str(Path("~/nilearn_data").expanduser()))
    return paths


def load_sample_motor_activation_image():
    """Load a single functional image showing motor activations.

    Returns
    -------
    str
        Path to the sample functional image.
    """
    from .func import load_sample_motor_activation_image as tmp

    warn(
        (
            f"{_GENERAL_MESSAGE}"
            "Please import this function from 'nilearn.datasets.func' instead."
        ),
        DeprecationWarning,
    )
    return tmp()
