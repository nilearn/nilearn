
"""
Simple Threshold Extractor
"""

import nibabel
import numpy as np

from scipy.ndimage import label
from scipy.stats import scoreatpercentile

from sklearn.base import BaseEstimator

from nilearn import plotting
from nilearn._utils import check_niimg
from nilearn.image import iter_img


def extract_connected_components(img, min_size):
    """ Extract the components which have largest
        connections given a thresholded image of the
        components

    Parameters
    ----------
    img: 3D/4D Nifti like object
        a thresholded image of the components

    Returns
    -------
    the extracted largest connected components of each thresholded
    map

    """
    components, labels = label(img)
    label_count = np.bincount(components.ravel())
    label_count[0] = 0
    for label_assign, count in enumerate(label_count):
        if count < min_size:
            img[components == label_assign] = 0

    return img


class SimpleThresholdExtractor(BaseEstimator):
    """ Finds the maximum value of the maps and picks only the
        voxels which have value more than half of the maximum value.

        Parameters
        ----------
        n_regions: int
            number of regions to extract from a set of brain maps

        min_size: int, optional
            size of the voxels for the components extraction

        Returns
        -------
        segmented regions of the brain maps
    """
    def __init__(self, n_regions=None, min_size=20,
                 ratio=1., threshold=None):
        self.n_regions = n_regions
        self.min_size = min_size
        self.ratio = ratio
        self.threshold = threshold

    def transform(self, maps_img):
        """ Extract the regions from the maps

        Parameters
        ----------
        maps_img: a Nifti like object or path to the filename

        Returns
        -------
        regions of the largest connected components
        """
        print self
        min_size = self.min_size
        maps_img = check_niimg(maps_img)
        maps_data = maps_img.get_data()
        # threshold will choose automaticaly based on max value of the data
        if isinstance(self.threshold, float):
            threshold = self.threshold
        elif self.threshold is None:
            threshold = max(abs(maps_data.min()), abs(maps_data.max()))
            threshold = threshold / 2
        elif self.threshold == 'auto':
            ratio = self.ratio
            abs_maps_data = np.abs(maps_data)
            threshold = scoreatpercentile(
                abs_maps_data,
                100. - (100. / len(maps_data)) * ratio)
        elif self.threshold is not None:
            raise ValueError("Threshold must be done, "
                             "'auto' or float. You provided %s." %
                             str(self.threshold))

        maps_affine = maps_img.get_affine()
        regions_img = []

        for img in iter_img(maps_img):
            img_data = img.get_data()

            img_data[img_data < threshold] = 0.
            region = extract_connected_components(img_data, min_size)
            regions_img.append(region)

        self.regions_ = regions_img


class VoxelRatioExtractor(SimpleThresholdExtractor):
    """ Finds threshold automatically and picks only voxels
        which are in the brain volume by multiplying the ratio
        with the number of voxels in the maps

    """
    def __init__(self, n_regions=None, min_size=20,
                 ratio=1., threshold='auto'):

        self.n_regions = n_regions
        self.ratio = ratio
        self.threshold = threshold
        self.min_size = min_size

    def transform(self, maps_img):
        """ Extract the regions from the maps

        Parameters
        ----------
        maps_img: Niimg like object or path to the filename

        Returns
        -------
        regions of the largest connected components
        """
        print self
        return super(VoxelRatioExtractor, self).transform(maps_img)
