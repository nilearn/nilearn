"""
region_signal_extractor
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
from nilearn._utils import check_niimg, check_niimg_3d, check_niimg_4d
from nilearn._utils.extmath import fast_abs_percentile
from nilearn.image import iter_img, new_img_like
from nilearn.image.image import _smooth_array
from nilearn._utils.niimg_conversions import concat_niimgs
from nilearn._utils.compat import _basestring


def apply_threshold_to_maps(maps, threshold, threshold_strategy):
    """ This function uses the specific type of threshold strategy
    and keeps the most prominent features of the maps which lies
    above a certain threshold.
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


def extract_regions(actual_map_data, process_map_data, min_size):
    """ This function takes the connected components of the
    maps data and automatically segments each component into
    a seperated region.

    Parameters
    ----------
    actual_map_data: a numpy array
        an original raw data array of the decomposed maps without
        being transformed to mask image space. This is generally
        used to get same voxel intensities as a raw data which quite
        helps in differentiating between negative and positive
        statistical values of the maps.

    process_map_data: a numpy array
        a data array same as actual_map_data which is further used
        for segementation processing.

    min_size: int
        An integer which actually limits the size of the regions to
        segment. Only the size of the voxels in the regions which
        are more than this number are selected and less than this
        number are ignored.

    Returns
    -------
    regions: a numpy array
        contains the data array of each extracted region appended in a
        one by one form for each of its input process data.
    """
    regions = []
    label_maps, n_labels = label(process_map_data)
    # Takes the size of each labelized region data
    labels_size = np.bincount(label_maps.ravel())
    labels_size[0] = 0.

    for label_id, label_size in enumerate(labels_size):
        if label_size > min_size:
            region_data = (label_maps == label_id) * actual_map_data
            regions.append(region_data)

    return regions


class region_signal_extractor(NiftiMapsMasker):
    """ Region Extraction is a post processing technique which
    is implemented to automatically segment each brain atlas maps
    into different set of seperated brain activated region.
    Particularly, to show that each decomposed brain maps can be
    used to focus on a target specific Regions of Interest analysis.

    Parameters
    ----------
    n_regions: int, default is None
        An integer which limits the number of regions to extract
        from the set of brain maps.

    mask: filename, Niimg, instance of NiftiMasker, default None, optional
        Mask to be used on data. If an instance of masker is passed,
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
        this value is replaced directly by the pre-defined set ratio 0.8
        in both cases of threshold estimation.

    threshold_strategy: string {'voxelratio', 'percentile'}, default is 'voxelratio', optional
        This parameter selects the way it applies the estimated threshold onto the data.
        If 'voxelratio' or 'percentile', the regions which are survived above an
        estimated threshold are kept as more intense foreground voxels to segment.

    extractor: string {'auto', 'local_regions'}, optional
        A string which selects between the type of extractor. If 'voxel_wise',
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
    def __init__(self, n_regions=None, mask=None, target_affine=None,
                 target_shape=None, standardize=False, low_pass=None,
                 high_pass=None, t_r=None, memory=Memory(cachedir=None),
                 min_size=20, threshold_strategy='voxelratio', threshold='auto',
                 extractor='auto', smooth_fwhm=6., verbose=0):
        self.n_regions = n_regions
        self.mask = mask
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.standardize = standardize
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.memory = memory

        self.min_size = min_size
        self.threshold_strategy = threshold_strategy
        self.threshold = threshold
        self.extractor = extractor
        self.smooth_fwhm = smooth_fwhm
        self.verbose = verbose

    def fit(self, maps_img):
        """ Compute the mask and fit the mask to the maps data and
        extract or seperate the regions into region from the maps.

        Parameters
        ----------
        maps_img: a Niimg-like image/object.
            the image which consists of atlas maps or statistically
            estimated maps.
        """
        maps_img = check_niimg_4d(maps_img)
        actual_maps_data = maps_img.get_data()
        self.maps_data = actual_maps_data
        affine = maps_img.get_affine()
        min_size = self.min_size

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
            self.masker_.fit(maps_img)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_
        # Maps which are transformed to mask image space and
        # therefore used for further processing simply denoted as "maps"
        maps = self.masker_.transform(maps_img)

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

        maps_thresholded = apply_threshold_to_maps(
            maps, self.threshold, self.threshold_strategy)

        all_regions_accumulated = []
        index_of_each_map = []
        all_regions_toimgs = []

        for index, map_process in enumerate(maps_thresholded):
            each_map_data = actual_maps_data[..., index]
            process_map_img = self.masker_.inverse_transform(map_process)
            process_map_data = process_map_img.get_data()

            if self.extractor == 'auto':
                regions_of_each_map = extract_regions(
                    each_map_data, process_map_data, min_size)
            elif self.extractor == 'local_regions':
                smooth_fwhm = self.smooth_fwhm
                smooth_data = _smooth_array(process_map_data,
                                            affine, fwhm=smooth_fwhm)
                seeds = peak_local_max(smooth_data, indices=False,
                                       exclude_border=False)
                seeds_label, seeds_id = label(seeds)
                # Assigning "-1" as ignored area to random walker
                seeds_label[process_map_data == 0] = -1
                max_value_process_data = np.max(process_map_data)
                # Values are normalized with max value to not exceed 1.
                if max_value_process_data > 1.:
                    process_map_data /= max_value_process_data
                seeds_map = random_walker(process_map_data, seeds_label,
                                          mode='cg_mg')
                # Again replace "-1" values with "0" for an expected behaviour
                # to region seperation
                seeds_map[seeds_map == -1] = 0
                regions_of_each_map = extract_regions(
                    each_map_data, seeds_map, min_size)

            len_regions_of_each_map = len(regions_of_each_map)
            index_of_each_map.extend([index] * len_regions_of_each_map)
            all_regions_accumulated.extend(regions_of_each_map)
        # Converting all regions which are accumulated to Nifti Image
        n_regions_accumulated = len(all_regions_accumulated)
        for n in range(n_regions_accumulated):
            each_region_toimg = new_img_like(maps_img,
                                             all_regions_accumulated[n])
            all_regions_toimgs.append(each_region_toimg)

        all_regions_toimgs = concat_niimgs(all_regions_toimgs)

        self.index_ = index_of_each_map
        self.regions_ = all_regions_toimgs
        return self

    def transform(self, imgs, confounds):
        """ Transform region signals to voxel timeseries signals.

        Parameters
        ----------
        imgs: a Niimg-like images/objects
            Data on which regions signals are transformed to voxel
            time series signals.

        confounds: CSV file path or 2D matrix
            This parameter cleans each subject data prior to timeseries
            signal extraction. It is recommended parameter especially
            in learning funtional connectomes in seperate brain regions.
            The most confounds such as high variance confounds or
            white matter or csf signals or motion regressors are regressed
            by passing to nilearn.signal.clean. Please see the related
            documentation for more details.

        Returns
        -------
        signals: a numpy array
            a list of averaged timeseries signals of each of the region.
        """
        signals = []
        if hasattr(self, "regions_"):
            regions_extracted_ = self.regions_
        else:
            raise ValueError('Regions are not extracted by calling fit(). '
                             'You must call fit() then followed by transform() '
                             'to extract the timeseries signals from those '
                             'extracted regions. ')

        nifti_maps_masker = NiftiMapsMasker(regions_extracted_,
                                            self.masker_.mask_img_)
        nifti_maps_masker.fit()
        for img, confound in zip(imgs, confounds):
            each_subject_signals = nifti_maps_masker.transform(
                img, confounds=confound)
            signals.append(each_subject_signals)

        return signals

