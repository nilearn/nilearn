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
from ..image import iter_img, new_img_like
from ..image.image import _smooth_array
from .._utils.niimg_conversions import concat_niimgs
from .._utils.compat import _basestring
from ..externals.skimage import peak_local_max, random_walker


def estimate_apply_threshold_to_maps(maps_img, mask_img=None, parameters=None,
                                     threshold='auto',
                                     estimate_threshold_value='percentile'):
    """ A function which keeps the most prominent regions of the maps. Can also
    be denoted as foreground objects/regions extraction.

    Parameters
    ----------
    maps_img: a 3D/4D Nifti like image/object
        a path or filename or an image consists of statistical maps.

    mask_img: a Nifti like image/object, default is None
        a path or filename or a mask image. To make sure that we keep regions
        which are within the brain volume.
        If given, maps are masked with this image or if not given, maps are
        masked automatically estimated by a `NiftiMasker`.

    parameters: list of parameters, default is None
        If given, parameters are passed to `filter_and_mask` along with
        the provided input mask image.

    threshold: a float (t-value, ratio), 'auto', default is 'auto'
        a value used to threshold the maps.

        If float, a t-statistic threshold value of maps, the intensities
        which are survived above this value are kept. Most suitable if user
        knows exactly the values to keep.

        If float, a ratio value between [0. and 1.], is used to multiply
        with the total size of voxels (ratio * n_voxels = some_voxels).
        The most intense voxels (some_voxels) which are survived will be kept.
        Most suitable for the case, if the concept of threshold should be
        according to total count of voxels inside the brain volume.

        If default 'auto', pre-defined ratio 0.8 is used, which means 80%
        of the voxels which are survived are kept.

    estimate_threshold_value: string {None, 'ratio_n_voxels', 'percentile'} \
        default 'percentile'
        a parameter to select between the type of determining a threshold value.
        It can be either based on number of voxels or score of the data based
        on a percentile.

        If None, threshold value will not be determined.

        If ratio_n_voxels, threshold value is estimated based on total size of the
        voxels.

        If percentile, threshold value is determined based on the percentage
        score of the input data of the maps.

    Returns
    -------
    threshold_maps_img: a Nifti like image/object
        a thresholded image of the input.
    """
    maps_img = check_niimg(maps_img)
    if len(maps_img.shape) > 3:
        single_map = False
    else:
        single_map = True

    if mask_img is not None:
        mask_img = check_niimg_3d(mask_img)
        if parameters is not None:
            maps, affine = filter_and_mask(maps_img, mask_img, parameters)
        else:
            nifti_masker = NiftiMasker(mask_img=mask_img)
            maps = nifti_masker.fit_transform(maps_img)
    elif mask_img is None:
        nifti_masker = NiftiMasker(mask_img=mask_img)
        maps = nifti_masker.fit_transform(maps_img)
        mask_img = nifti_masker.mask_img_

    ratio = None
    if isinstance(threshold, float):
        if estimate_threshold_value is None:
            # When threshold is needed to apply directly based on
            # statistical values which should not be more than maximum
            value_check = abs(maps).max()
            if abs(threshold) > value_check:
                raise ValueError("The value given to threshold "
                                 "statistical maps must not exceed %d. "
                                 "You provided threshold=%s " % (value_check,
                                                                 threshold))
            else:
                cutoff_threshold = threshold
        else:
            ratio = threshold
    elif threshold == 'auto':
        ratio = 0.8
    elif threshold is not None:
        raise ValueError("Threshold must be either "
                         "'auto' or float. You provided %s." %
                         str(threshold))
    # check if the input strategy is a valid
    list_of_estimators = [None, 'percentile', 'ratio_n_voxels']
    if estimate_threshold_value not in list_of_estimators:
        message = ("'threshold_strategy' should be given as "
                   "either of these {0}").format(list_of_estimators)
        raise ValueError(message)

    # Thresholding
    if ratio is not None and ratio > 1.:
        raise ValueError("threshold given for a 'percentile' or 'ratio_n_voxels' "
                         "is expected between 0. and 1. "
                         "You provided %s" % threshold)

    if estimate_threshold_value == 'percentile':
        percentile = 100. - (100. / len(maps)) * ratio
        cutoff_threshold = scoreatpercentile(np.abs(maps), percentile)
    elif ratio is not None and estimate_threshold_value == 'ratio_n_voxels':
        raveled = np.abs(maps).ravel()
        argsort = np.argsort(raveled)
        n_voxels = (ratio * maps.size)
        cutoff_threshold = raveled[argsort[- n_voxels]]

    maps[np.abs(maps) < cutoff_threshold] = 0.
    threshold_maps = maps
    threshold_maps_img = masking.unmask(threshold_maps, mask_img)
    # squeeze the image if input maps given as 3D is converted to
    # 4D after thresholding
    if single_map:
        data = threshold_maps_img.get_data()
        affine = threshold_maps_img.get_affine()
        threshold_maps_img = new_img_like(
            threshold_maps_img, data[:, :, :, 0], affine)

    return threshold_maps_img


def break_connected_components(map_img, min_size=20,
                               extract_type='connected_components',
                               peak_local_smooth=6, mask_img=None):
    """ A function takes the connected components of the brain activation
    maps/regions and breaks each component into a seperate brain regions.

    Parameters
    ----------
    map_img: a Nifti-like image/object
        a 3D image of the activation maps which should be breaked into a set
        of regions.

    min_size: int, default is 20
        An integer which denotes the size of voxels in the each region.
        Only the size of the regions which are more than this number
        are kept.

    extract_type: string {"connected_components", "local_regions"} \
        default is connected_components
        A method used to segment/seperate the regions.

        If connected_components, each component in the input image is assigned
        a unique label point and then seperated based on the uniqueness of those
        label points.

        If local_regions, smoothing followed by random walker procedure is
        used to split into each a seperate regions.

    peak_local_smooth: scalar, default is 6mm
        a value in mm which is used to smooth an image to locate seed points.

    mask_img: Nifti-like image/object, default is None
        an option used to mask the input brain map image.

    Returns
    -------
    regions_accumulated: a Nifti-like images
        contains the images of segmented regions each 3D image appended as a
        seperate brain activated region.
    """
    regions_accumulated = []
    map_img = check_niimg(map_img)
    if len(map_img.shape) == 0 or len(map_img.shape) == 4:
        raise ValueError('A 3D Nifti image or path to a 3D image should '
                         'be submitted.')

    extract_methods = ['connected_components', 'local_regions']
    if extract_type not in extract_methods:
        message = ("'extract_type' should be given "
                   "either of these {0}").format(extract_methods)
        raise ValueError(message)

    map_data = map_img.get_data()
    affine = map_img.get_affine()
    # Mark the seeds using random walker
    if extract_type == 'local_regions':
        smooth_map_data = _smooth_array(map_data, affine, peak_local_smooth)
        seeds = peak_local_max(smooth_map_data, indices=False,
                               exclude_border=False)
        seeds_label, seeds_id = label(seeds)
        # Assign integer "-1" to ignore
        seeds_label[map_data == 0.] = -1
        # Take the maximum value of the data and
        # normalize to be between -1 and 1
        max_value_map = max(map_data.max(), -map_data.min())
        if max_value_map > 1.:
            map_data /= max_value_map
        rw_maps = random_walker(map_data, seeds_label, mode='cg_mg')
        # Now simply replace "-1" with "0" for regions seperation
        rw_maps[rw_maps == -1] = 0.
        maps_assigned = rw_maps
    else:
        # Taking directly the input data which is the case in 'auto' type
        maps_assigned = map_data
    # Region seperation
    label_maps, n_labels = label(maps_assigned)
    # Takes the size of each labelized region data
    labels_size = np.bincount(label_maps.ravel())
    labels_size[0] = 0.
    for label_id, label_size in enumerate(labels_size):
        if label_size > min_size:
            region_data = (label_maps == label_id) * map_data
            region_img = new_img_like(map_img, region_data)
            regions_accumulated.append(region_img)

    return regions_accumulated


class RegionExtractor(NiftiMapsMasker):
    """ Class to decompose maps into seperated brain regions.

    Region Extraction is a post processing technique which
    is implemented to automatically segment each brain atlas maps
    into different set of seperated brain activated region.
    Particularly, to show that each decomposed brain maps can be
    used to focus on a target specific Regions of Interest analysis.

    Parameters
    ----------
    maps_img: Niimg-like object or path to the Niimg
       an image or a filename of the image which contains a set of brain
       atlas maps or statistically decomposed brain maps.

    mask: Niimg-like object, instance of NiftiMasker, default is None, optional
        Mask to be applied on the input data. If an instance of masker is passed,
        then its mask is used. If no mask is provided, this class/function
        will automatically computes mask by NiftiMasker from the input data.

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

    threshold: float or 'auto', default is 'auto', optional
        If default 'auto', we estimate a threshold with pre-defined ratio set as 0.8.
        The estimated threshold tells us to keep the more intense voxels within
        the brain volume across all maps. The threshold value is estimated by
        multiplying the ratio with total size of the voxels, denoted as
        (0.8 * n_voxels) or it can be estimated by multiplying the ratio with
        the total number of maps to get in a percentile.

        If given as float, this value is replaced directly by the pre-defined
        set ratio 0.8.

    threshold_strategy: string {'ratio_n_voxels', 'percentile'}, \
        default is 'ratio_n_voxels', optional
        This parameter is to select between the type of estimating the threshold.

        If 'ratio_n_voxels', the threshold value is estimated based on total
        size of voxels in the brain volume. This option tells us to keep the
        percentage of more intense voxels.

        If 'percentile', the percentile threshold value is estimated by the
        percentage of the score to the number of maps.

    extractor: string {'connected_components', 'local_regions'}, optional
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
    `regions_`: Niimg-like image/object
        a list of seperated regions with each region lying on a 3D volume
        concatenated into a 4D Nifti like object.

    `index_`: a numpy array
        an array of list of indices where each index value is assigned to
        each seperate region of its corresponding family of brain maps.

    `signals_`: a numpy array
        a list of averaged timeseries signals of the subjects extracted from
        each region.

    References
    ----------
    * Abraham et al. "Region segmentation for sparse decompositions: better
      brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
      Sep 2014, Boston, United States. pp.8
    """
    def __init__(self, maps_img, mask=None, target_affine=None,
                 target_shape=None, standardize=False, low_pass=None,
                 high_pass=None, t_r=None, memory=Memory(cachedir=None),
                 min_size=20, threshold_strategy='ratio_n_voxels',
                 threshold='auto', extractor='connected_components',
                 peak_local_smooth=6., verbose=0):
        self.maps_img = maps_img
        self.mask = mask
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.standardize = standardize
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory

        # parameters for region extraction
        self.min_size = min_size
        self.threshold_strategy = threshold_strategy
        self.threshold = threshold
        self.extractor = extractor
        self.peak_local_smooth = peak_local_smooth

        self.verbose = verbose

    def fit(self, X=None, y=None):
        """ Prepare or set up the data for the region extraction

        """
        self.maps_img_ = check_niimg(self.maps_img)

        if isinstance(self.mask, NiftiMasker):
            self.masker_ = clone(self.mask)
        else:
            self.masker_ = NiftiMasker(mask_img=self.mask,
                                       target_affine=self.target_affine,
                                       target_shape=self.target_shape,
                                       standardize=False,
                                       low_pass=self.low_pass,
                                       high_pass=self.high_pass,
                                       mask_strategy='background',
                                       t_r=self.t_r,
                                       memory=self.memory)

        if self.masker_.mask_img is None:
            self.masker_.fit(self.maps_img_)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        parameters = get_params(NiftiMasker, self)
        parameters['detrend'] = True
        # foreground extraction of input data "maps_img"
        self.threshold_maps_img_ = estimate_apply_threshold_to_maps(
            self.maps_img_, mask_img=self.mask_img_, parameters=parameters,
            threshold=self.threshold, estimate_threshold_value=self.threshold_strategy)

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
        if not hasattr(self, 'threshold_maps_img_') \
                and not hasattr(self, 'mask_img_'):
            message = ("It seems like either threshold_maps_img or mask_img "
                       " is missing. You must call fit() before calling a "
                       "transform(). or You must call fit_transform() directly")
            raise ValueError(message)

        all_regions_accumulated = []
        index_of_each_map = []
        all_regions_toimgs = []
        region_signals = []

        for index, map_ in enumerate(iter_img(self.threshold_maps_img_)):
            regions_imgs_of_each_map = break_connected_components(
                map_, self.min_size, self.extractor, self.peak_local_smooth)
            len_regions_of_each_map = len(regions_imgs_of_each_map)
            index_of_each_map.extend([index] * len_regions_of_each_map)
            all_regions_accumulated.extend(regions_imgs_of_each_map)

        all_regions_toimgs = concat_niimgs(all_regions_accumulated)
        regions_extracted = all_regions_toimgs

        nifti_maps_masker = NiftiMapsMasker(regions_extracted,
                                            self.mask_img_,
                                            standardize=self.standardize)
        nifti_maps_masker.fit()
        if confounds is None:
            confounds = [None] * len(imgs)
        for img, confound in zip(imgs, confounds):
            each_subject_signals = nifti_maps_masker.transform(
                img, confounds=confound)
            region_signals.append(each_subject_signals)

        self.index_ = index_of_each_map
        self.regions_ = regions_extracted
        self.signals_ = region_signals

        return self

