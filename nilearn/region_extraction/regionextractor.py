"""
region_signal_extractor
"""

import nibabel
import numpy as np

from scipy.ndimage import label
from scipy.stats import scoreatpercentile

from skimage.feature import peak_local_max
from skimage.segmentation import random_walker

from nilearn import plotting
from nilearn._utils import check_niimg
from nilearn._utils.extmath import fast_abs_percentile
from nilearn.image import iter_img, new_img_like
from nilearn.image.image import _smooth_array
from nilearn._utils.niimg_conversions import concat_niimgs
from nilearn._utils.compat import _basestring


def extract_regions(img, labels_data, min_size):
    """ Extract each region from the connected components
        data where the data is already assigned with a
        unique label point.

    Parameters
    ----------
    label_data: a numpy array
        an array of data where each data point is labelled.

    Returns
    -------
    regions_img: a 4D Nifti like image/object
        contains the regions extracted from
        the labelled data. Each extracted region
        lies in a list of 3D volumes.
    """
    regions = []
    img_data = img.get_data()

    labels_size = np.bincount(labels_data.ravel())
    labels_size[0] = 0.

    for label_id, label_size in enumerate(labels_size):
        if label_size > min_size:
            region_img = new_img_like(img,
                                      (labels_data == label_id) * img_data)
            regions.append(region_img)

    return regions


class region_signal_extractor(object):
    """ Region Extraction is a post processing technique which
    is implemented to automatically extract each brain atlas maps
    into different set of seperated brain activated region.
    Particularly, to show that each decomposed brain maps can be
    used to focus on a target specific Regions of Interest analysis.

    Parameters
    ----------
    n_regions: int, default is None
        limit the number of regions to extract from a set of
        brain maps.

    min_size: int, default size 20, optional
        suppresses the smallest spurious regions by selecting
        the regions with the size of voxels survived more than
        "min_size".

    threshold: float or default string {'auto'}, optional
        chooses to extract the foreground objects. If float,
        this value is directly used as ratio multiplied to the
        "scoreatpercentile" of each map data to get highly
        reasonable threshold value. We take values of the map
        which have more than threshold value. If 'auto', the
        strategy is same as seen in float case but here
        difference is that ratio is predefined to 1. We
        recommend to use default which promises to extract
        reasonable foreground objects.

    extractor: {'voxel_regions', 'local_regions'}, optional
        switches between which technique to be used. If
        'voxel_regions', regions are first thresholded to
        foreground objects and then labels are assigned to
        each objects to get the end result. If 'local_regions',
        same strategy follows as in 'voxel_regions' but the
        labels are assigned after each region is detected by
        its own peak max value technically called as seed points.

    smooth_fwhm: scalar, default smooth_fwhm=6. optional
        smoothing parameter as a full width half maximum,
        in millimetres. This scalar value is applied on
        all three directions.

    Returns
    -------
    regions_: a 4D Nifti like image/object
        a list of seperate regions with each region lying on
        a 3D volume concatenated into a 4D Nifti object.

    index_: a numpy array
        an array of list of indices where each index value is
        assigned to each seperate region of its corresponding
        family of brain maps.

    References
    ----------
    * Abraham et al. "Region segmentation for sparse decompositions: better
      brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
      Sep 2014, Boston, United States. pp.8
    """
    def __init__(self, n_regions=None, min_size=20,
                 threshold='auto',
                 extractor='voxel_regions',
                 smooth_fwhm=6., verbose=0):
        self.n_regions = n_regions
        self.min_size = min_size
        self.threshold = threshold
        self.extractor = extractor
        self.smooth_fwhm = smooth_fwhm
        self.verbose = verbose

    def transform(self, maps_img):
        """ Extract the regions from the maps

        Parameters
        ----------
        maps_img: a Nifti like image/object or the filename
            the image consists of atlas maps or statistical maps

        Returns
        -------
        regions_: 4D Nifti like image/object
            a list of seperate regions concatenated in a 4D Nifti
            like image.

        index_: a numpy array
            a list of indices in an array to get the identity of each
            region to a family of brain atlas maps.
        """
        min_size = self.min_size
        maps_img = check_niimg(maps_img)
        len_maps = maps_img.shape[3]

        extractor_methods = ['voxel_regions', 'local_regions']
        if self.extractor not in extractor_methods:
            message = ('"extractor" should be given '
                       'either of these {0}').format(extractor_methods)
            raise ValueError(message)

        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 1.
        elif self.threshold is not None:
            raise ValueError("Threshold must be, "
                             "'auto' or float. You have given %s."
                             % str(self.threshold))

        regions_accumulated = []
        index = []

        for i, img in enumerate(iter_img(maps_img)):
            data = img.get_data()
            affine = img.get_affine()
            each_map = data.copy()

            percentile = 100 - (100 / len_maps) * ratio
            threshold = scoreatpercentile(
                np.abs(data), percentile)
            each_map[np.abs(data) < threshold] = 0.

            if self.extractor == 'voxel_regions':
                label_maps, n_labels = label(each_map)
                regions = extract_regions(img, label_maps, min_size)
                regions_accumulated.append(regions)
            elif self.extractor == 'local_regions':
                smooth_fwhm = self.smooth_fwhm
                smooth_data = _smooth_array(data,
                                            affine, fwhm=smooth_fwhm)
                seeds = peak_local_max(smooth_data, indices=False,
                                       exclude_border=False)
                seeds_label, seeds_id = label(seeds)
                # Assign "-1" as ignored area to random walker
                seeds_label[each_map == 0] = -1
                seeds_map = random_walker(each_map, seeds_label,
                                          mode='cg_mg')
                seeds_map[seeds_map == -1] = 0
                seeds_label_maps, n_seeds_labels = label(seeds_map)
                regions = extract_regions(img, seeds_label_maps, min_size)
                regions_accumulated.append(regions)

        all_regions_imgs = concat_niimgs(regions_accumulated)
        index.extend([i] * all_regions_imgs.shape[3])

        self.index_ = index
        self.regions_ = all_regions_imgs

