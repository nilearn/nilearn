"""
Downloading NeuroImaging datasets: resting-state network templates
"""
import os
import xml.etree.ElementTree
import numpy as np

from sklearn.datasets.base import Bunch
from sklearn.utils import deprecated

from .utils import _get_dataset_dir, _fetch_files, _get_dataset_descr

from .._utils import check_niimg
from ..image import new_img_like
from .._utils.compat import _basestring, get_affine


def fetch_allen_rsn_tmap_75_2011(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Allen fMRI resting-state network 
    templates published in E. Allen, et al, Frontiers in Systems Neuroscience, 2011.

    The provided images are in MNI152 space.

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    url: string
        url of file to download.

    resume: bool
        whether to resumed download of a partly-downloaded file.

    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:
            all_unthresh_tmaps
            all_unthresh_rsn_labels
            rsn_unthresh_tmaps
            aggregate_ic_comps

    References
    ----------
    Licence: unknown

    E. Allen, et al, "A baseline for the multivariate comparison of resting
    state networks," Frontiers in Systems Neuroscience, vol. 5, p. 12, 2011.

    See http://mialab.mrn.org/data/index.html for more information
    on this dataset.
    """
    if url is None:
        url = "http://mialab.mrn.org/data/hcp/"

    dataset_name = "allen_rsn_2011"
    keys = ("all_unthresh_tmaps",
            "rsn_unthresh_tmaps",
            "aggregate_ic_comps")

    opts = {}
    files = ["ALL_HC_unthresholded_tmaps.nii",
             "RSN_HC_unthresholded_tmaps.nii",
             "rest_hcp_agg__component_ica_.nii",
            ]

    filenames = [(f, url + f, opts) for f in files]

    package_directory = os.path.dirname(os.path.abspath(__file__))
    labels = os.path.join(package_directory, "data",
                          "ALL_HC_unthresholded_tmaps.txt")

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    params = dict([('description', fdescr),
                   ('all_unthresh_rsn_labels', labels)] +
                   list(zip(keys, sub_files)))

    return Bunch(**params)
