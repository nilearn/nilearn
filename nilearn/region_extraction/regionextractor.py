"""
region_extractor
"""

import nibabel
import numpy as np

from scipy.ndimage import label

from skimage.feature import peak_local_max
from skimage.segmentation import random_walker

from nilearn import plotting
from nilearn._utils import check_niimg
from nilearn._utils.extmath import fast_abs_percentile
from nilearn.image import iter_img, new_img_like
from nilearn.image.image import _smooth_array
from nilearn._utils.niimg_conversions import concat_niimgs
from nilearn._utils.compat import _basestring


def extract_seperate_regions(img, img_data, min_size):
    """ Extract and seperate the regions which have largest
        connections considering group of voxels which are
        certainly high given a thresholded image of the
        brain maps/components.

    Parameters
    ----------
    img: 3D Nifti like image/object
        contains a thresholded image of the components.

    Returns
    -------
    regions_img: a 4D Nifti like image/object
        contains the regions extracted from
        the input image where each 3D volume is
        assigned as each seperate region by
        concatenating in the 4th dimension to a 4D image.
    """
    regions_img = []
    affine = img.get_affine()
    region = img_data.copy()

    labels, n_labels = label(img_data)
    labels_size = np.bincount(labels.ravel())
    labels_size[0] = 0.

    for label_id, label_size in enumerate(labels_size):
        if label_size > min_size:
            region = img_data.copy()
            region = (labels == label_id) * img_data
            region_img = new_img_like(img, region, affine)
            regions_img.append(region_img)

    return regions_img


class RegionExtractor():
    """ Region Extraction is a post processing technique which
    is implemented to automatically extracts each brain atlas maps
    into different set of seperated brain activated region.
    Particularly, to show that each decomposed brain maps can be
    used to focus on a target specific Regions of Interest analysis.
    The idea to implement is a two step procedure:
        1. Foreground Extraction using automatic thresholding function
            based upon "percentile" of the features.
            Ref: "fast_abs_percentile" in nilearn._utils.extmath
        2. Biggest connected components extraction and assigning each
            component a label to seperate out into each different region.
            Ref: "label" in scipy.ndimage
    Based upon the idea, two types of region extraction techniques
    are designed:
        1. "Auto Extractor" is a simple mechanism which is a way to
            follows a two step procedure.
        2. "Random Walker Extractor" in addition to a two step procedure
            RWE is a way more robust mechanism than "Auto Extractor".
            It goes this way which assigns labels using seed markers by
            assigning each label with a seed point by using a standard
            "peak_local_max" from scikit image.
            Ref: "peak_local_max" in skimage.measure
            Ref: "random_walker" in skimage.segmentation

    Parameters
    ----------
    n_regions: int, default is None
        limit the number of regions to extract from a set of brain maps.

    min_size: int, default size 20, optional
        restricts or suppresses the smallest spurious regions by selecting
        the regions within which have voxels survived more than "min_size".

    threshold: float or string, default string {'auto'}, optional
        chooses to extract the foreground objects. If float, this value is
        directly taken as a threshold or 'auto' option chooses automatically.
        We recommend to use default which promises to extract reasonable
        foreground objects.

    percentile: int, range [0 100], default=80, optional
        specifically used in threshold strategy which selects features
        based on percentile value.

    extractor: {'autoextractor', 'randomwalker'}, optional
        switches between which technique is to be used. If 'autoextractor',
        regions are extracted using a "Auto Extractor" method. Otherwise
        if 'randomwalker', regions are extracted using "Random Walker"
        segmentation algorithm. Both ideas are briefly defined above.

    seed_ratio: float or string, default string {'auto'}. optional

    smooth_fwhm: scalar, default smooth_fwhm=6. optional
        smoothing parameter as a full width half maximum, in millimetres.
        this scalar value is applied on all three directions.

    Returns
    -------
    regions_: a 4D Nifti like image/object
        a list of seperate regions with each region lying on a 3D volume
        concatenated into a 4D Nifti object.

    index_: a numpy array
        an array of list of indices where each index value is assigned to
        each seperate region of its corresponding family of brain maps.

    References
    ----------
    * Abraham et al. "Region segmentation for sparse decompositions: better
      brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
      Sep 2014, Boston, United States. pp.8
    """
    def __init__(self, n_regions=None, min_size=20,
                 threshold='auto', percentile=80,
                 extractor='autoextractor', seed_ratio='auto',
                 smooth_fwhm=6., verbose=0):
        self.n_regions = n_regions
        self.min_size = min_size
        self.threshold = threshold
        self.percentile = percentile
        self.extractor = extractor
        self.seed_ratio = seed_ratio
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

        extractor_methods = ['autoextractor', 'randomwalker']
        if (not isinstance(self.extractor, _basestring) or
                self.extractor not in extractor_methods):
            message = ('"extractor" should be given '
                       'either of these {0}').format(extractor_methods)
            raise ValueError(message)

        if isinstance(self.threshold, float):
            threshold = self.threshold
        elif self.threshold == 'auto':
            pass
        elif self.threshold is not None:
            raise ValueError("Threshold must be done, "
                             "'auto' or float. You provided %s." %
                             str(self.threshold))

        regions_imgs = []
        index = []
        percentile = self.percentile

        if self.extractor == 'randomwalker':
            randomwalker = True
            if isinstance(self.seed_ratio, float):
                seed_ratio = self.seed_ratio
            elif self.seed_ratio == 'auto':
                pass
            elif self.seed_ratio is not None:
                raise ValueError("seeds_ratio must be, "
                                 "'auto' or float. You have given %s." %
                                 str(self.seed_ratio))
        else:
            randomwalker = False

        for i, img in enumerate(iter_img(maps_img)):
            data = img.get_data()
            affine = img.get_affine()
            each_map = data.copy()

            if self.threshold == 'auto':
                threshold = fast_abs_percentile(each_map[each_map != 0].ravel(),
                                                percentile)
                # I have no specific motivation for this tolerance value
                # I used the same line from "find_xyz_cut_coords"
                threshold_each_map = np.abs(each_map) > threshold - 1.e-15
                label_map = threshold_each_map.copy()

            if randomwalker:
                if self.seed_ratio == 'auto':
                    smooth_fwhm = self.smooth_fwhm
                    smooth_data = _smooth_array(data, affine, fwhm=smooth_fwhm)
                    seeds = peak_local_max(smooth_data, indices=False,
                                           exclude_border=False)
                    seeds_label, seeds_id = label(seeds)
                    # Assign "-1" as ignored area to random walker
                    seeds_label[label_map == 0] = -1
                    label_map = random_walker(label_map, seeds_label,
                                              mode='cg_mg')
                    label_map[label_map == -1] = 0

            regions = extract_seperate_regions(img, label_map,
                                               min_size)
            regions_imgs.append(regions)
            all_regions_imgs = concat_niimgs(regions_imgs)
            index.extend([i] * all_regions_imgs.shape[3])

        self.index_ = index
        self.regions_ = all_regions_imgs

