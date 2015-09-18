"""
Better brain parcellations for Region of Interest analysis
"""

import nibabel
import numpy as np

from scipy.ndimage import label
from scipy.stats import scoreatpercentile

from sklearn.base import clone
from sklearn.externals.joblib import Memory

from skimage.feature import peak_local_max
from skimage.segmentation import random_walker

from nilearn import plotting
from nilearn.input_data import NiftiMasker, NiftiMapsMasker
from nilearn.input_data.nifti_masker import filter_and_mask
from nilearn._utils import check_niimg, check_niimg_3d, check_niimg_4d
from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.class_inspect import get_params
from nilearn.image import iter_img, new_img_like
from nilearn.image.image import _smooth_array
from nilearn._utils.niimg_conversions import concat_niimgs
from nilearn._utils.compat import _basestring


def apply_threshold_to_maps(maps, threshold, threshold_strategy):
    """ A function uses the specific strategy of threshold to keep the
    prominent features of the maps which lies above a certain threshold value.

    Parameters
    ----------
    maps: a numpy array
        a data consists of statistical maps or atlas maps.
    threshold: an integer
        a value which is used to threshold the maps.
    threshold_strategy: string {"voxelratio", "percentile"}
        a strategy which is used to select the way threshold should be done.

    Returns
    -------
    threshold_maps: a numpy array
        a thresholded maps in a numpy array format.
    """
    abs_maps = np.abs(maps)
    ratio = None
    if isinstance(threshold, float):
        ratio = threshold
    elif threshold == 'auto':
        ratio = 1.
    elif threshold is not None:
        raise ValueError("Threshold must be None, "
                         "'auto' or float. You provided %s." %
                         str(threshold))
    if ratio is not None and threshold_strategy == 'percentile':
        percentile = 100. - (100. / len(maps)) * ratio
        cutoff_threshold = scoreatpercentile(abs_maps, percentile)
    elif ratio is not None and threshold_strategy == 'voxelratio':
        raveled = abs_maps.ravel()
        argsort = np.argsort(raveled)
        n_voxels = (ratio * maps.size)
        cutoff_threshold = raveled[argsort[- n_voxels]]
    maps[abs_maps < cutoff_threshold] = 0.
    threshold_maps = maps

    return threshold_maps


def extract_regions(map_img, min_size, extract_type, smooth_fwhm, mask_img=None):
    """ This function takes the connected components which lies in a 3D brain
    map and automatically segments each component into a seperate region.

    Parameters
    ----------
    map_img: a Nifti-like image/object
        a 3D image of the activation maps which should be segmented.
    min_size: int
        An integer which denotes the size of voxels in the regions.
        Only the size of the regions which are more than this number
        are kept.
    extract_type: string {"auto", "local_regions"}
        If "auto", each unique component in the image is assigned with each
        a unique label point and then decomposed each into a seperate region.
        If "local_regions", smoothing is applied to the image and peak maximum
        value of each unique component is assigned with a unique seed points by
        random walker procedure and then decomposed into each assigned into a
        each a seperate region.
    smooth_fwhm: scalar
        a value in millimetres which is used to smooth an image to locate seed
        points.
    mask_img: Nifti-like image/object, default is None, optional
        an option used to mask the input brain map image.

    Returns
    -------
    regions: a Nifti-like images
        contains the images of segmented regions each 3D image appended as a
        seperate brain activated image.
    """
    regions_accumulated = []
    map_data = map_img.get_data()
    affine = map_img.get_affine()
    # Mark the seeds using random walker
    if extract_type == 'local_regions':
        smooth_map_data = _smooth_array(map_data, affine, smooth_fwhm)
        seeds = peak_local_max(smooth_map_data, indices=False,
                               exclude_border=False)
        seeds_label, seeds_id = label(seeds)
        # Assign integer "-1" to ignore
        seeds_label[map_data == 0.] = -1
        # Take the maximum value of the data and
        # normalize to be between -1 and 1
        max_value_map = np.max(map_data)
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

    n_regions: int, default is None
        An integer which limits the number of regions to extract from the
        set of brain maps.

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
        If 'auto', we estimate a threshold with pre-defined ratio set as 0.8.
        The estimated threshold tells us to keep the more intense voxels within
        the brain volume across all maps. The threshold value is estimated by
        multiplying the ratio with total size of the voxels, denoted as
        (0.8 * n_voxels) or it can be estimated by multiplying the ratio with
        the total number of maps to get in a percentile. If given as float,
        this value is replaced directly by the pre-defined set ratio 0.8 and
        the same is applied which is there is in no need of estimation.

    threshold_strategy: string {'voxelratio', 'percentile'}, default is 'voxelratio', optional
        This parameter selects the way it applies the estimated threshold onto the data.
        If 'voxelratio' or 'percentile', the regions which are survived above an
        estimated threshold are kept as more intense foreground voxels to segment.

    extractor: string {'auto', 'local_regions'}, optional
        A string which selects between the type of extractor. If 'auto',
        regions are segmented using labelling assignment to each unique object.
        If 'local_regions', regions are segmented using a seed points assigned by
        its peak max value of that particular local regions and then labelling
        assignment to each unique seed points.

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

    References
    ----------
    * Abraham et al. "Region segmentation for sparse decompositions: better
      brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
      Sep 2014, Boston, United States. pp.8
    """
    def __init__(self, maps_img, n_regions=None, mask=None, target_affine=None,
                 target_shape=None, standardize=False, low_pass=None,
                 high_pass=None, t_r=None, memory=Memory(cachedir=None),
                 min_size=20, threshold_strategy='voxelratio', threshold='auto',
                 extractor='auto', smooth_fwhm=6., verbose=0):
        self.maps_img = maps_img
        self.n_regions = n_regions
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
        self.smooth_fwhm = smooth_fwhm

        self.verbose = verbose

    def fit(self, X=None, y=None):
        """ Prepare or set up the data for the region extraction

        """
        self.maps_img_ = check_niimg_4d(self.maps_img)

        if isinstance(self.mask, NiftiMasker):
            self.masker_ = clone(self.mask)
        else:
            self.masker_ = NiftiMasker(mask_img=self.mask,
                                       target_affine=self.target_affine,
                                       target_shape=self.target_shape,
                                       standardize=self.standardize,
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

        threshold_strategy = ['voxelratio', 'percentile']
        if self.threshold_strategy not in threshold_strategy:
            message = ("'threshold_strategy' should be given "
                       "either of these {0}").format(threshold_strategy)
            raise ValueError(message)

        extractor_methods = ['auto', 'local_regions']
        if self.extractor not in extractor_methods:
            message = ('"extractor" should be given '
                       'either of these {0}').format(extractor_methods)
            raise ValueError(message)

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
            a list of averaged timeseries signals extracted from each region.

        """
        if hasattr(self, 'maps_img_') and hasattr(self, 'mask_img_'):
            parameters = get_params(NiftiMasker, self)
            parameters['detrend'] = True
            maps, affine = filter_and_mask(
                self.maps_img_, self.mask_img_, parameters)
            maps_threshold = apply_threshold_to_maps(
                maps, self.threshold, self.threshold_strategy)
        else:
            raise ValueError('Images of the maps are missing. '
                             'You must load the images by calling fit() '
                             'followed by a transform() or '
                             'call fit_transform() directly.')
        maps_threshold_img = self.masker_.inverse_transform(maps_threshold)

        all_regions_accumulated = []
        index_of_each_map = []
        all_regions_toimgs = []
        region_signals = []

        for index, map_ in enumerate(iter_img(maps_threshold_img)):
            regions_imgs_of_each_map = extract_regions(
                map_, self.min_size, self.extractor, self.smooth_fwhm)
            len_regions_of_each_map = len(regions_imgs_of_each_map)
            index_of_each_map.extend([index] * len_regions_of_each_map)
            all_regions_accumulated.extend(regions_imgs_of_each_map)

        all_regions_toimgs = concat_niimgs(all_regions_accumulated)
        regions_extracted = all_regions_toimgs

        nifti_maps_masker = NiftiMapsMasker(regions_extracted,
                                            self.masker_.mask_img_)
        nifti_maps_masker.fit()
        for img, confound in zip(imgs, confounds):
            each_subject_signals = nifti_maps_masker.transform(
                img, confounds=confound)
            region_signals.append(each_subject_signals)

        self.index_ = index_of_each_map
        self.regions_ = regions_extracted
        self.signals_ = region_signals

        return self

