"""
Better brain parcellations for Region of Interest analysis
"""

import nibabel
import numpy as np

from scipy.ndimage import label
from scipy.stats import scoreatpercentile

from sklearn.base import clone
from sklearn.externals.joblib import Memory

from .. import masking
from ..input_data import NiftiMasker, NiftiMapsMasker
from ..input_data.nifti_masker import filter_and_mask
from .._utils import check_niimg, check_niimg_3d, check_niimg_4d
from .._utils.extmath import fast_abs_percentile
from .._utils.class_inspect import get_params
from ..image import iter_img, new_img_like, resample_img
from ..image.image import _smooth_array
from .._utils.niimg_conversions import concat_niimgs, _check_same_fov
from .._utils.compat import _basestring
from ..externals.skimage import peak_local_max, random_walker


def foreground_extraction(maps_img, mask_img=None, threshold='auto',
                          thresholding_strategy='percentile'):
    """ A function which keeps the most prominent regions of the maps,
    denoted as foreground objects/regions extraction.

    Parameters
    ----------
    maps_img: a 3D/4D Nifti like image/object
        a path or filename or an image consists of statistical maps or atlas
        maps.

    mask_img: a Nifti like image/object, default is None
        If given, mask image is applied directly on the input data 'maps_img'.
        For example, if interested to consider only one particular brain region
        then mask image to that particular brain region should be provided.
        If none, masking will not be applied.

    threshold: a float (t-value or ratio [0., 100.]), 'auto', default is 'auto'
        a value used to threshold the maps to keep the most meaningful voxels.

        If float representing as a t-value, a t-statistic value should be submitted
        to keep the voxel intensities which are above than this value.
        Mostly suitable, if user wants to keep the voxels based on raw statistic value.
        In this case, user should know exactly which are the voxels to be kept by
        visually checking the intensity value.

        If float representing as a ratio, this value is used to keep the more intense
        voxels across all maps. The most meaningful voxels are kept by multiplying
        this value with the total size of the voxels.

        If default 'auto', a pre-defined float value set to 0.8 will be used
        in representing as a ratio.

    thresholding_strategy: string {None, 'ratio_n_voxels', 'percentile'}, \
        default 'percentile'.
        a string used to select between two different types of thresholding strategies.
        This strategy takes the given threshold value and thresholds the data.

        If ratio_n_voxels, most meaningful voxels which are survived above the
        value which is determined by ratio * n_voxels will be kept.

        If percentile, most meaningful voxels which are survived above the
        value determined by (percentile = ratio * 100) of the total voxels.

        If None, neither of the strategy is used which means no thresholding will
        be done.

    Returns
    -------
    threshold_maps_img: a Nifti like image/object
        a thresholded image of the input.
    """
    maps_img = check_niimg(maps_img, atleast_4d=True)
    maps = maps_img.get_data()
    len_of_maps = maps.shape[-1]
    affine = maps_img.get_affine()

    if mask_img is not None:
        if not _check_same_fov(maps_img, mask_img):
            mask_img = resample_img(mask_img,
                                    target_affine=maps_img.get_affine(),
                                    target_shape=maps_img.shape[:3],
                                    interpolation="nearest")

        mask = masking._load_mask_img(mask_img)

        # Set as 0 for the values which are outside of the mask
        maps[mask == 0.] = 0.

    list_of_strategies = [None, 'percentile', 'ratio_n_voxels']
    if thresholding_strategy not in list_of_strategies:
        message = ("'thresholding_strategy' should be given as "
                   "either of these {0}").format(list_of_strategies)
        raise ValueError(message)

    ratio = None
    if isinstance(threshold, float):
        if thresholding_strategy is None:
            # When threshold is needed to apply directly based on
            # statistical values, the given value should not be more than maximum
            value_check = abs(maps).max()
            if abs(threshold) > value_check:
                raise ValueError("The value given to threshold "
                                 "statistical maps must not exceed %d. "
                                 "You provided threshold=%s " % (value_check,
                                                                 threshold))
            cutoff_threshold = threshold
        else:
            ratio = threshold
    elif threshold == 'auto' and ratio is None:
        ratio = 0.8
    elif threshold is not None:
        raise ValueError("Threshold must be None "
                         "'auto' or float. You provided %s." %
                         str(threshold))
    # Thresholding
    if ratio is not None and thresholding_strategy == 'percentile':
        percentile = 100. - (100. / len_of_maps) * ratio
        cutoff_threshold = scoreatpercentile(np.abs(maps), percentile)
    elif ratio is not None and thresholding_strategy == 'ratio_n_voxels':
        raveled = np.abs(maps).ravel()
        argsort = np.argsort(raveled)
        n_voxels = (ratio * maps.size)
        cutoff_threshold = raveled[argsort[- n_voxels]]

    maps[np.abs(maps) < cutoff_threshold] = 0.
    threshold_maps = maps

    threshold_maps_img = new_img_like(maps_img, threshold_maps, affine)

    return threshold_maps_img


def connected_component_extraction(maps_img, min_size=20,
                                   extract_type='local_regions',
                                   peak_local_smooth=6, mask_img=None):
    """ A function takes the connected components of the brain activation
    maps/regions and breaks each component into a seperate brain regions.

    Parameters
    ----------
    maps_img: a Nifti-like image/object
        an image of the activation or atlas maps which should be extracted
        into a set of regions.

    min_size: int, default is 20
        An integer which denotes the size of voxels in the each region.
        Only the size of the regions which are more than this number
        are kept.

    extract_type: string {"connected_components", "local_regions"} \
        default is local_regions
        A method used to segment/seperate the regions.

        If connected_components, each component in the input image is assigned
        a unique label point and then seperated based on only label points.

        If local_regions, smoothing followed by random walker procedure is
        used to split into each a seperate regions.

    peak_local_smooth: scalar, default is 6mm
        a value in mm which is used to smooth an image to locate seed points.

    mask_img: Nifti-like image/object, default is None
        If given, mask image is applied directly on the input data 'maps_img'.
        For example, if interested to consider only one particular brain region
        then mask image to that particular brain region should be provided.
        If none, masking will not be applied.

    Returns
    -------
    regions_extracted: a 4D Nifti-like image
        contains the images of segmented regions each 3D image is a
        seperate brain activated region or atlas region.

    indentity_id_of_maps: a numpy array
        an array of list of indices where each index value is assigned to
        each seperate region of its corresponding family of brain maps.
    """
    all_regions_imgs = []
    identity_id_of_maps = []
    maps_img = check_niimg(maps_img, atleast_4d=True)
    maps = maps_img.get_data()
    affine = maps_img.get_affine()

    extract_methods = ['connected_components', 'local_regions']
    if extract_type not in extract_methods:
        message = ("'extract_type' should be given "
                   "either of these {0}").format(extract_methods)
        raise ValueError(message)

    if mask_img is not None:
        if not _check_same_fov(maps_img, mask_img):
            mask_img = resample_img(mask_img,
                                    target_affine=maps_img.get_affine(),
                                    target_shape=maps_img.shape[:3],
                                    interpolation="nearest")
            mask = masking._load_mask_img(mask_img)
            # Set as 0 to the values which are outside of the mask
            maps[mask == 0.] = 0.

    for index in range(maps.shape[-1]):
        regions = []
        map_ = maps[..., index]
        # Mark the seeds using random walker
        if extract_type == 'local_regions':
            smooth_map = _smooth_array(map_, affine, peak_local_smooth)
            seeds = peak_local_max(smooth_map, indices=False,
                                   exclude_border=False)
            seeds_label, seeds_id = label(seeds)
            # Assign -1 to values which are 0. to indicate to ignore
            seeds_label[map_ == 0.] = -1
            # Take the maximum value of the data and
            # normalize to be between -1 and 1 for random walker compatibility
            max_value_map = max(map_.max(), -map_.min())
            if max_value_map > 1.:
                map_ /= max_value_map

            try:
                mode = 'cg_mg'
                rw_maps = random_walker(map_, seeds_label, mode=mode)
            except Exception as e:
                print "Random Walker algorithm failed for mode:%s, %s" % (mode, str(e))
                rw_maps = random_walker(map_, seeds_label, mode='bf')

            # Now simply replace "-1" with "0" for regions seperation
            rw_maps[rw_maps == -1] = 0.
            maps_assigned = rw_maps
        else:
            # Taking directly the input data which is the case in 'auto' type
            maps_assigned = map_

        # Region seperation block
        label_maps, n_labels = label(maps_assigned)
        # Takes the size of each labelized region data
        labels_size = np.bincount(label_maps.ravel())
        labels_size[0] = 0.
        for label_id, label_size in enumerate(labels_size):
            if label_size > min_size:
                region_data = (label_maps == label_id) * map_
                region_img = new_img_like(maps_img, region_data)
                regions.append(region_img)

        identity_id_of_maps.extend([index] * len(regions))
        all_regions_imgs.extend(regions)
        del regions
    regions_extracted = concat_niimgs(all_regions_imgs)

    return regions_extracted, identity_id_of_maps


class RegionExtractor(NiftiMapsMasker):
    """ Class to decompose maps into seperated brain regions.

    Region Extraction is a post processing technique which
    is implemented to automatically segment each brain atlas maps
    into different set of seperated brain activated region.
    Particularly, to show that each decomposed brain maps can be
    used to focus on a target specific Regions of Interest analysis.

    Parameters
    ----------
    maps_img: 4D Niimg-like object or path to the 4D Niimg
       an image or a filename of the image which contains a set of brain
       atlas maps or statistically decomposed brain maps.

    mask_img: Niimg-like object or None,  default is None, optional
        Mask to be applied on the input data.
        If given, mask_img is submitted to an instance of NiftiMapsMasker to
        for a mask preparation to input data.
        If None, no mask will be applied on the data.

    target_affine: 3x3 or 4x4 matrix, default is None, optional
        If given, while masking the image is resampled to this affine by
        passing the parameter to image.resample_img.
        Please see the related documentation for more details or
        http://nilearn.github.io/manipulating_visualizing/data_preparation.html

    target_shape: 3-tuple of integers, default is None, optional
        If given, while masking the image is resized to match the given shape.
        This parameter is passed to image.resample_img.
        Please see the related documentation for more details or
        http://nilearn.github.io/manipulating_visualizing/data_preparation.html

    standardize: boolean, True or False, default True, optional
        If True, the time series signals are centered and normalized by
        putting their mean to 0 and variance to 1. Recommended to
        set as True if signals are not already standardized.

    low_pass: float, default is None, optional
        This value will be applied on the signals by passing to signal.clean
        Please see the related documentation signal.clean for more details.

    high_pass: float, default is None, optional
        This value will be applied on the signals by passing to signal.clean
        Please see the related documentation signal.clean for more details.

    t_r: float, default is None, optional
        Repetition time in sec. This value is given to signal.clean
        Please see the related documentation for details.

    memory: instance of joblib.Memory, string, default is None, optional
        Used to cache the masking process. If a string is given, the path
        is set with this string as a folder name in the directory.

    min_size: int, default size 20, optional
        An integer which suppresses the smallest spurious regions by selecting
        the regions having the size of the voxels which are more than this
        integer.

    threshold: float (t-value or ratio [0., 100.]), 'auto', default is 'auto'
        a value used to threshold the maps to keep the most meaningful voxels.
        If float representing as a t-value, a t-statistic value should be submitted
        to keep the voxel intensities which are above than this value.
        Mostly suitable, if user wants to keep the voxels based on raw statistic value.
        In this case, user should know exactly which are the voxels to be kept by
        visually checking the intensity value.

        If float representing as a ratio, this value is used to keep the more intense
        voxels across all maps. The most meaningful voxels are kept by multiplying
        this value with the total size of the voxels.

        If default 'auto', a pre-defined float value set to 0.8 will be used
        in representing as a ratio.

    threshold_strategy: string {'ratio_n_voxels', 'percentile'}, \
        default is 'percentile', optional
        a string used to select between two different types of thresholding strategies.
        This strategy takes the given threshold value and thresholds the data.

        If ratio_n_voxels, most meaningful voxels which are survived above the
        value which is determined by ratio * n_voxels will be kept.

        If percentile, most meaningful voxels which are survived above the
        value determined by (percentile = ratio * 100) of the total voxels will be kept.

    extractor: string {'connected_components', 'local_regions'}, optional
        default is 'local_regions'
        A string which chooses between the type of extractor.

        If 'connected_components', regions are segmented using only a labels
        assigned to each unique region/object.

        If 'local_regions', regions are segmented using a seed points assigned by
        its peak max value of that particular local regions.

    smooth_fwhm: scalar, default smooth_fwhm=6. optional
        smoothing parameter as a full width half maximum, in millimetres.
        This scalar value is applied on all three directions (x,y,z).

    Attributes
    ----------
    regions_: Niimg-like image/object
        a list of seperated regions with each region lying on a 3D volume
        concatenated into a 4D Nifti like object.

    indentity_: a numpy array
        an array of list of indices where each index value is assigned to
        each seperate region of its corresponding family of brain maps.

    signals_: a numpy array
        a list of averaged timeseries signals of the subjects extracted from
        each region.

    References
    ----------
    * Abraham et al. "Region segmentation for sparse decompositions: better
      brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
      Sep 2014, Boston, United States. pp.8
    """
    def __init__(self, maps_img, mask_img=None, target_affine=None,
                 target_shape=None, standardize=False, low_pass=None,
                 high_pass=None, t_r=None, memory=Memory(cachedir=None),
                 min_size=20, thresholding_strategy='percentile',
                 threshold='auto', extractor='local_regions',
                 peak_local_smooth=6., verbose=0):
        self.maps_img = maps_img
        self.mask_img = mask_img
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.standardize = standardize
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory

        # parameters for region extraction
        self.min_size = min_size
        self.thresholding_strategy = thresholding_strategy
        self.threshold = threshold
        self.extractor = extractor
        self.peak_local_smooth = peak_local_smooth

        self.verbose = verbose

    def fit(self, X=None, y=None):
        """ Prepare or set up the data for the region extraction

        """
        maps_img = check_niimg_4d(self.maps_img)

        # Asking NiftiMapsMasker to prepare the data for regions extraction
        # only if mask_img is provided
        if self.mask_img is not None:
            nifti_maps_masker = NiftiMapsMasker(maps_img, mask_img=self.mask_img,
                                                resampling_target="maps")
            nifti_maps_masker.fit()
            self.maps_img_ = nifti_maps_masker.maps_img_
            self.mask_img_ = nifti_maps_masker.mask_img_
        else:
            self.mask_img_ = None
            self.maps_img_ = maps_img

        # foreground extraction
        self.threshold_maps_img_ = foreground_extraction(
            self.maps_img_,
            mask_img=self.mask_img_,
            threshold=self.threshold,
            thresholding_strategy=self.thresholding_strategy)

        # connected component extraction
        all_regions = []
        index_each_map = []
        self.regions_, self.identity_ = connected_component_extraction(
            self.threshold_maps_img_,
            self.min_size,
            self.extractor,
            self.peak_local_smooth)

        return self

    def fit_transform(self, imgs, confounds=None):
        return self.fit().transform(imgs, confounds=confounds)

    def transform(self, imgs, confounds=None):
        """ Extract the region time series signals from the list
        of 4D Nifti like images.

        Parameters
        ----------
        imgs: 4D Nifti-like images/objects.
            Data on which region signals are transformed to voxel
            timeseries signals.

        confounds: CSV file path or 2D matrix, optional
            This parameter cleans each subject data prior to region
            signals extraction. It is recommended parameter especially
            in learning functional connectomes in brain regions. The
            most common confounds such as high variance or white matter or
            csf signals or motion regressors are regressed out by passsing
            to nilearn.signal.clean. Please see the related documentation
            for more details.

        Returns
        -------
        region_signals: a numpy array
            an averaged time series signals of the subjects.
        """
        region_signals = []
        if not hasattr(self, 'regions_') \
                and not hasattr(self, 'mask_img_'):
            message = ("It seems like either extracted regions or mask_img "
                       " is missing. You must call fit() before calling a "
                       "transform(). or You must call fit_transform() directly")
            raise ValueError(message)

        nifti_maps_masker = NiftiMapsMasker(self.regions_,
                                            mask_img=self.mask_img_,
                                            standardize=self.standardize)
        nifti_maps_masker.fit()
        if confounds is None:
            confounds = [None] * len(imgs)
        for img, confound in zip(imgs, confounds):
            each_subject_signals = nifti_maps_masker.transform(
                img, confounds=confound)
            region_signals.append(each_subject_signals)

        self.signals_ = region_signals

        return self

