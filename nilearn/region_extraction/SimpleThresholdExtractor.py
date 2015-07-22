"""
Voxel_ratio_extractor
"""

import nibabel
import numpy as np

from scipy.ndimage import label
from scipy.stats import scoreatpercentile

from nilearn import plotting
from nilearn._utils import check_niimg
from nilearn._utils.niimg_conversions import concat_niimgs
from nilearn.image import iter_img, new_img_like


def extract_seperate_regions(img, img_data, affine, min_size):
    """ Extract the components which have largest
        connections given a thresholded image of the
        components

        Parameters
        ----------
        img: numpy array of 3D Nifti like object
            contains a thresholded image of the components

        Returns
        -------
        the seperated or segmented regions extracted from
        the input image which contains a set of connected
        maps

    """
    all_regions_img = []
    # n_components and n_labels before removing
    # spurious regions
    components, labels = label(img_data)
    label_count = np.bincount(components.ravel())
    label_count[0] = 0
    for label_assign, count in enumerate(label_count):
        if count < min_size:
            img_data[components == label_assign] = 0

    # Extract n_regions and n_labels survived after
    # minimizing the unwanted or spurious regions
    n_components, n_regions = label(img_data)
    for n in range(n_regions):
        region = (n_components == n) * img_data
        region_img = new_img_like(img, region, affine)
        all_regions_img.append(region_img)
        regions_4dfile = concat_niimgs(all_regions_img)

    return regions_4dfile


class VoxelRatioExtractor():
    """ Finds threshold automatically by choosing the
        number of voxels which more intense voxels within
        the brain volume.

        Parameters
        ----------
        n_regions: int
            number of regions to extract from a set
            of brain maps

        min_size: int, optional
            size of the voxels for the regions extraction

        Returns
        -------
        segmented regions of the brain maps
    """
    def __init__(self, n_regions=None, min_size=20,
                 ratio=1., threshold='auto'):
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
        all_imgs_4d = []
        index = []

        for i, img in enumerate(iter_img(maps_img)):
            img_data = img.get_data()

            img_data[img_data < threshold] = 0.
            imgs_4dfile = extract_seperate_regions(img, img_data,
                                                   maps_affine, min_size)

            all_imgs_4d.append(imgs_4dfile)
            all_regions_extracted = concat_niimgs(all_imgs_4d)
            index.extend([i] * all_regions_extracted.shape[3])

        self.index_ = index
        self.regions_ = all_regions_extracted
