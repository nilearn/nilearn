"""
Better brain parcellations for Region of Interest analysis
"""
import warnings
import numbers
import nibabel
import numpy as np

from scipy.ndimage import label
from scipy.stats import scoreatpercentile

from sklearn.base import clone
from sklearn.externals.joblib import Memory

from .. import masking
from ..input_data import NiftiMapsMasker
from .._utils import check_niimg, check_niimg_4d
from ..image import new_img_like, resample_img
from ..image.image import _smooth_array, threshold_img
from .._utils.niimg_conversions import concat_niimgs, _check_same_fov
from .._utils.compat import _basestring
from .._utils.ndimage import _peak_local_max
from .._utils.segmentation import _random_walker


def _threshold_maps(maps_img, threshold):
    """ Automatic thresholding of atlas maps image.

    Considers the given threshold as a ratio to the total number of voxels
    in the brain volume. This gives a certain number within the data
    voxel size which means that nonzero voxels which fall above than this
    size will be kept across all the maps.

    Parameters
    ----------
    maps_img: Niimg-like object
        an image of brain atlas maps.
    threshold: float
        value used as a ratio to n_voxels to get a certain threshold size in
        number to threshold the image. The value should be positive and within
        the range of number of maps (i.e. n_maps in 4th dimension).

    Returns
    -------
    threshold_maps_img: Nifti1Image object
        gives us thresholded image.
    """
    maps = check_niimg(maps_img)
    n_maps = maps.shape[-1]
    if not isinstance(threshold, float) or threshold <= 0 or threshold > maps.shape[-1]:
        raise ValueError("threshold given as ratio to the number of voxels must "
                         "be float value and should be positive and between 0 and "
                         "total number of maps i.e. n_maps={0}. "
                         "You provided {1}".format(maps.shape[-1], threshold))
    else:
        ratio = threshold

    maps_data = maps.get_data()
    abs_maps = np.abs(maps_data)
    # thresholding
    cutoff_threshold = scoreatpercentile(
        abs_maps, 100. - (100. / n_maps) * ratio)
    maps_data[abs_maps < cutoff_threshold] = 0.

    threshold_maps_img = new_img_like(maps, maps_data)

    return threshold_maps_img


def connected_regions(maps_img, min_region_size=50, extract_type='local_regions',
                      smoothing_fwhm=6, mask_img=None):
    """ Extraction of brain connected regions into separate regions.

    Parameters
    ----------
    maps_img: Niimg-like object
        an image of brain activation or atlas maps to be extracted into set of
        separate brain regions.

    min_region_size: int, default 50, optional
        Minimum number of voxels for a region to be kept. Useful to suppress
        small spurious regions.

    extract_type: str {'connected_components', 'local_regions'} \
        default local_regions, optional
        If 'connected_components', each component/region in the image is extracted
        automatically by labelling each region based upon the presence of unique
        features in their respective regions.
        If 'local_regions', each component/region is extracted based on their
        maximum peak value to define a seed marker and then using random walker
        segementation algorithm on these markers for region separation.

    smoothing_fwhm: scalar, default 6mm, optional
        To smooth an image to extract most sparser regions. This parameter
        is passed `_smooth_array` and exists only for extract_type 'local_regions'.

    mask_img: Niimg-like object, default None
        If given, mask image is applied to input data.
        If None, no masking is applied.

    Returns
    -------
    regions_extracted_img: Nifti1Image object
        gives the image in 4D of extracted brain regions. Each 3D image consists
        of only one separated region.

    index_of_each_map: numpy array
        an array of list of indices where each index denotes the identity
        of each extracted region to their family of brain maps.
    """
    all_regions_imgs = []
    index_of_each_map = []
    maps_img = check_niimg(maps_img, atleast_4d=True)
    maps = maps_img.get_data()
    affine = maps_img.get_affine()

    allowed_extract_types = ['connected_components', 'local_regions']
    if extract_type not in allowed_extract_types:
        message = ("'extract_type' should be given either of these {0} "
                   "You provided extract_type='{1}'").format(allowed_extract_types, extract_type)
        raise ValueError(message)

    if mask_img is not None:
        if not _check_same_fov(maps_img, mask_img):
            mask_img = resample_img(mask_img,
                                    target_affine=maps_img.get_affine(),
                                    target_shape=maps_img.shape[:3],
                                    interpolation="nearest")
            mask_data, _ = masking._load_mask_img(mask_img)
            # Set as 0 to the values which are outside of the mask
            maps[mask_data == 0.] = 0.

    for index in range(maps.shape[-1]):
        regions = []
        map_3d = maps[..., index]
        # Mark the seeds using random walker
        if extract_type == 'local_regions':
            smooth_map = _smooth_array(map_3d, affine=affine, fwhm=smoothing_fwhm)
            seeds = _peak_local_max(smooth_map)
            seeds_label, seeds_id = label(seeds)
            # Assign -1 to values which are 0. to indicate to ignore
            seeds_label[map_3d == 0.] = -1
            rw_maps = _random_walker(map_3d, seeds_label)
            # Now simply replace "-1" with "0" for regions seperation
            rw_maps[rw_maps == -1] = 0.
            label_maps = rw_maps
        else:
            # Connected component extraction
            label_maps, n_labels = label(map_3d)

        # Takes the size of each labelized region data
        labels_size = np.bincount(label_maps.ravel())
        # set background labels sitting in zero index to zero
        labels_size[0] = 0.
        for label_id, label_size in enumerate(labels_size):
            if label_size > min_region_size:
                region_data = (label_maps == label_id) * map_3d
                region_img = new_img_like(maps_img, region_data)
                regions.append(region_img)

        index_of_each_map.extend([index] * len(regions))
        all_regions_imgs.extend(regions)

    regions_extracted_img = concat_niimgs(all_regions_imgs)

    return regions_extracted_img, index_of_each_map


class RegionExtractor(NiftiMapsMasker):
    """ Class for brain region extraction.

    Region Extraction is a post processing technique which
    is implemented to automatically segment each brain atlas maps
    into different set of separated brain activated region.
    Particularly, to show that each decomposed brain maps can be
    used to focus on a target specific Regions of Interest analysis.

    Parameters
    ----------
    maps_img: 4D Niimg-like object
       Image containing a set of whole brain atlas maps or statistically
       decomposed brain maps.

    mask_img: Niimg-like object or None,  default None, optional
        Mask to be applied to input data, passed to NiftiMapsMasker.
        If None, no masking is applied.

    min_region_size: int, default 50, optional
        Minimum number of voxels for a region to be kept. Useful to suppress
        small spurious regions.

    threshold: float or str, default string "80%", optional
        If it is a float, it will be used in ratio_n_voxels threshold strategy.
        If string, it should finish with percent sign e.g. "80%" and will be
        used in percentile threshold strategy.

    thresholding_strategy: str {'percentile', 'ratio_n_voxels'},\
        default 'percentile', optional
        If default 'percentile', images are thresholded based on the percentage
        of the score on the data and the scores which are survived above this
        percentile will be kept.
        If set to 'ratio_n_voxels', meaning we keep the more intense brain voxels
        n_voxels across all maps. The probability of chance of nonzero voxels
        survived after taking ratio to the total number of brain voxels will be kept.
        For example, more the voxels to be kept high should be the ratio within the
        total number of maps.

    extractor: str {'connected_components', 'local_regions'} default 'local_regions', optional
        If 'connected_components', each component/region in the image is extracted
        automatically by labelling each region based upon the presence of unique
        features in their respective regions.
        If 'local_regions', each component/region is extracted based on their
        maximum peak value to define a seed marker and then using random walker
        segementation algorithm on these markers for region separation.

    standardize: bool, True or False, default False, optional
        If True, the time series signals are centered and normalized by
        putting their mean to 0 and variance to 1. Recommended to
        set as True if signals are not already standardized.
        passed to class NiftiMapsMasker.

    detrend: bool, True or False, default False, optional
        This parameter is passed to nilearn.signal.clean basically
        indicates whether to detrend timeseries signals or not.
        passed to class NiftiMapsMasker.

    low_pass: float, default None, optional
        This value will be applied on the signals by passing to signal.clean
        Please see the related documentation signal.clean for more details.
        passed to class NiftiMapsMasker.

    high_pass: float, default None, optional
        This value will be applied on the signals by passing to signal.clean
        Please see the related documentation signal.clean for more details.
        passed to NiftiMapsMasker.

    t_r: float, default None, optional
        Repetition time in sec. This value is given to signal.clean
        Please see the related documentation for details.
        passed to NiftiMapsMasker.

    memory: instance of joblib.Memory, string, default None, optional
        Used to cache the masking process. If a string is given, the path
        is set with this string as a folder name in the directory.
        passed to NiftiMapsMasker.

    memory_level: int, default 0, optional
        Aggressiveness of memory catching. The higher the number, the higher
        the number of functions that will be cached. Zero mean no caching.
        passed to NiftiMapsMasker.

    verbose: int, default 0, optional
        Indicates the level of verbosity by printing the message. Zero indicates
        nothing is printed.

    Attributes
    ----------
    regions_img_: Nifti1Image object
        list of separated regions with each region lying on a 3D volume
        concatenated into a 4D image.

    index_: numpy array
        array of list of indices where each index value is assigned to
        each separate region of its corresponding family of brain maps.

    References
    ----------
    * Abraham et al. "Region segmentation for sparse decompositions: better
      brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
      Sep 2014, Boston, United States. pp.8
    """
    def __init__(self, maps_img, mask_img=None, min_region_size=50,
                 threshold="80%", thresholding_strategy='percentile',
                 extractor='local_regions', standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0, verbose=0):
        super(RegionExtractor, self).__init__(
            maps_img=maps_img, mask_img=mask_img,
            standardize=standardize, detrend=detrend, low_pass=low_pass,
            high_pass=high_pass, t_r=t_r, memory=memory,
            memory_level=memory_level, verbose=verbose)
        self.maps_img = maps_img
        self.min_region_size = min_region_size
        self.thresholding_strategy = thresholding_strategy
        self.threshold = threshold
        self.extractor = extractor

    def fit(self, X=None, y=None):
        """ Prepare the data and setup for the region extraction
        """
        maps_img = check_niimg_4d(self.maps_img)

        # foreground extraction
        if self.thresholding_strategy == 'ratio_n_voxels':
            if not isinstance(self.threshold, float):
                raise ValueError("threshold should be given as float value "
                                 "for thresholding_strategy='ratio_n_voxels'. "
                                 "You provided a value of threshold={0}".format(self.threshold))
            threshold_maps = _threshold_maps(maps_img, self.threshold)
        else:
            threshold_maps = threshold_img(maps_img, mask_img=self.mask_img,
                                           threshold=self.threshold,
                                           thresholding_strategy=self.thresholding_strategy)

        # connected component extraction
        self.regions_img_, self.index_ = connected_regions(threshold_maps,
                                                           self.min_region_size,
                                                           self.extractor)

        self.maps_img = self.regions_img_
        super(RegionExtractor, self).fit()

        return self
