"""
Auto Extractor
"""

import numpy as np

from scipy.ndimage.measurements import label

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

from nilearn._utils import check_niimg


def compute_low_threshold(maps, ratio):
    """
    Parameters
    ----------
    maps: numpy array like of
        a 3D/4D Niimg-like object which contains data of the
        input maps

    ratio: a float value
        this value is used to apply on the maps

    Returns
    -------
    a lower threshold value computed from the maps

    """
    if not isinstance(maps, np.ndarray):
        raise ValueError("Given input maps should be an array like")

    maps_ravel = np.abs(maps).ravel()
    argsort = np.argsort(maps_ravel)
    n_voxels = int(ratio * maps[0].size)
    low_threshold = maps_ravel[argsort[- n_voxels]]
    return low_threshold


def compute_regions(maps):
    """
    Parameters
    ----------
    maps: 3D/4D Nifti like object

    Returns
    -------
    the regions which are strongly connected components
    from the maps

    """
    label_maps = []
    for map_ in maps:
        if np.all(map_ == 0):
            continue
        label_maps.append(label(map_)[0])
    return label_maps


class VoxelRatioExtractor(BaseEstimator, TransformerMixin):
    """ Automatically finds the threshold value of the maps and
        takes the biggest connected components of the ICA maps
        or any output from decomposition technique.

        Parameters
        ----------
        n_regions: int
            number of regions to extract from a set of
            3D brain maps obtained by a decomposition method.
            For example, method such as ICA or MSDL.

        only_gm: a Nifti like object, optional
            a grey matter template to extract regions only
            from the grey matter. By default, ICBM gm template.

        value_ratio: a float value, optional
            this value is set to compute the low value on the maps
            which are then discarded as a spurious regions.

        min_size: int, optional
            to determine the minimum size of the voxels survived within
            the regions.

        Attributes
        ----------
        regions_img_: a Nifti like object
            returns a 4D Nifti object where each 3D image contains the
            labels of the regions extracted from the input maps.

        References
        ---------
        * Abraham et al. "Region segmentation for sparse decompositions: better
        brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
        Sep 2014, Boston, United States. pp.8
    """
    def __init__(self, value_ratio=1., min_size=10, n_regions=None,
                 only_gm=False):
        super(VoxelRatioExtractor, self).__init__(
                value_ratio=value_ratio, min_size=min_size,
                n_regions=n_regions, only_gm=only_gm)
        self.value_ratio = value_ratio
        self.min_size = min_size
        self.n_regions = n_regions
        self.only_gm = only_gm

    def transform(self, maps_img):
        """ Extract and assing the labels to the regions from the maps

            Parameters
            ----------
            maps_img: Nifti like object or path of the images
                a 4D image which contains regions extracted lying on each of the
                3D image. A 4D image has a set of brain regions to be extracted.

            Returns
            -------
            returns assigned labels to further extend to the connected components
        """
        maps_img = check_niimg(maps_img)
        maps = maps_img.get_data()
        maps = np.rollaxis(maps.view(), 3)

        self.affine = maps_img.get_affine()

        if self.value_ratio is not None:
            ratio = self.value_ratio
            low_threshold = compute_low_threshold(maps, ratio)
            maps[np.abs(maps) < low_threshold] = 0.

        regions = compute_regions(maps)

