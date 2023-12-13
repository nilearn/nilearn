"""Downloading NeuroImaging datasets: utility functions."""
import os
from pathlib import Path

from .._utils import fill_doc


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

        paths.append(os.path.expanduser("~/nilearn_data"))
    return paths


def load_sample_motor_activation_image():
    """Load a single functional image showing motor activations.

    Returns
    -------
    str
        Path to the sample functional image.
    """
    return str(Path(__file__).parent / "data" / "image_10426.nii.gz")
