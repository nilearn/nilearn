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


def extract_regions(map_data, affine, label_data, min_size):
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

    labels_size = np.bincount(label_data.ravel())
    labels_size[0] = 0.

    for label_id, label_size in enumerate(labels_size):
        if label_size > min_size:
            region = (label_data == label_id) * map_data
            region_img = new_img_like(nibabel.Nifti1Image(map_data, affine),
                                      region)
            regions.append(region_img)

    return regions


class region_signal_extractor(NiftiMapsMasker):
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

    mask: filename, Niimg, instance of NiftiMasker, default None, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask is used. If no mask is provided, this function
        will automatically compute mask from NiftiMasker.

    target_affine: 3x3 or 4x4 matrix, default is None, optional
        If given, while masking the image is resampled to this
        affine by passing the parameter to image.resample_img.
        Please see the related documentation for more details or
        http://nilearn.github.io/manipulating_visualizing/data_preparation.html

    target_shape: 3-tuple of integers, default is None, optional
        If given, while masking the image is resized to match
        the given shape. This parameter is passed to image.resample_img.
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
        suppresses the smallest spurious regions by selecting
        the regions with the size of voxels survived more than
        "min_size".

    threshold_strategy: string {'voxelratio', 'percentile'}, optional
        This parameter is designed to choose the way threshold should
        be calculated.
        If 'voxelratio', threshold is calculated by taking the more intense
        number of voxels (n_voxels) multiplied by threshold value or 'ratio'.
        If 'percentile', threshold is calculated by scoring using a percentile
        function.

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

    extractor: {'voxel_wise', 'local_regions'}, optional
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

    Attributes
    ----------
    `regions_`: Niimg-like image/object
        a list of seperate regions with each region lying on
        a 3D volume concatenated into a 4D Nifti object.

    `index_`: a numpy array
        an array of list of indices where each index value is
        assigned to each seperate region of its corresponding
        family of brain maps.

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
                 extractor='voxel_wise', smooth_fwhm=6., verbose=0):
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
        and extract or seperate the regions into region from the maps.

        Parameters
        ----------
        maps_img: a Niimg-like image/object.
            the image which consists of atlas maps or statistically
            estimated maps.

        Returns
        -------
        regions_: 4D Niimg-like image/objects.
            a list of seperated region converted to Niimg concatenated
            in a 4th dimension.

        index_: a numpy array
            a list of indices in an array to see the identity of each
            seperated region to a family of atlas maps.
        """
        maps_img = check_niimg(maps_img)
        len_maps = maps_img.shape[:3]
        maps_data = maps_img.get_data()
        self.maps_data = maps_data
        affine = maps_img.get_affine()
        min_size = self.min_size
        if isinstance(self.mask, (_basestring, nibabel.Nifti1Image)):
            mask = self.mask
            mask_img = check_niimg_3d(mask)
            if maps_img.shape[:3] != mask_img.shape:
                raise ValueError("Provided mask image 'mask' and "
                                 "3D shape of input maps must have "
                                 "identical shape")
            else:
                self.masker_.mask_img_ = mask
        elif isinstance(self.mask, NiftiMasker):
            self.masker_ = clone(self.mask)
        elif self.mask is None:
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
        self.transform_maps_ = self.masker_.transform(maps_img)
        transformed_maps = self.transform_maps_

        threshold_strategy = ['voxelratio', 'percentile']
        if self.threshold_strategy not in threshold_strategy:
            message = ('"threshold_stratefgy" should be given '
                       'either of these {0}').format(threshold_strategy)
            raise ValueError(message)

        extractor_methods = ['voxel_wise', 'local_regions']
        if self.extractor not in extractor_methods:
            message = ('"extractor" should be given '
                       'either of these {0}').format(extractor_methods)
            raise ValueError(message)

        if isinstance(self.threshold, float):
            ratio = self.threshold
        elif self.threshold == 'auto':
            ratio = 0.8
        elif self.threshold is not None:
            raise ValueError('Threshold must be given as '
                             '"auto" or float. You have given %s. '
                             % str(self.threshold))

        if self.threshold_strategy == 'voxelratio':
            raveled = np.abs(transformed_maps).ravel()
            argsort = np.argsort(raveled)
            n_voxels = (ratio * transformed_maps.size)
            threshold = raveled[argsort[- n_voxels]]
            transformed_maps[np.abs(transformed_maps) < threshold] = 0.
        elif self.threshold_strategy == 'percentile':
            percentile = 100 - (100 / len_maps) * ratio
            threshold = scoreatpercentile(
                np.abs(transformed_maps), percentile)
            transformed_maps[np.abs(transformed_maps) < threshold] = 0.

        regions_accumulated = []
        index = []

        for i, trans_map in enumerate(transformed_maps):
            each_map_data = maps_data[..., i]
            trans_map_img = self.masker_.inverse_transform(trans_map)
            trans_map_data = trans_map_img.get_data()

            if self.extractor == 'voxel_wise':
                label_maps, n_labels = label(trans_map_data)
                regions = extract_regions(each_map_data, affine,
                                          label_maps, min_size)
                regions_accumulated.append(regions)
            elif self.extractor == 'local_regions':
                smooth_fwhm = self.smooth_fwhm
                smooth_data = _smooth_array(trans_map_data,
                                            affine, fwhm=smooth_fwhm)
                seeds = peak_local_max(smooth_data, indices=False,
                                       exclude_border=False)
                seeds_label, seeds_id = label(seeds)
                # Assigning "-1" as ignored area to random walker
                seeds_label[trans_map_data == 0] = -1
                seeds_map = random_walker(trans_map_data, seeds_label,
                                          mode='cg_mg')
                # Again replace "-1" values with "0" for an expected behaviour
                # to region seperation
                seeds_map[seeds_map == -1] = 0
                seeds_label_maps, n_seeds_labels = label(seeds_map)
                regions = extract_regions(each_map_data, affine,
                                          seeds_label_maps, min_size)
                regions_accumulated.append(regions)

        all_regions_imgs = concat_niimgs(regions_accumulated)
        index.extend([i] * all_regions_imgs.shape[3])

        self.index_ = index
        self.regions_ = all_regions_imgs
        return self

    def transform(self, imgs):
        """ Transform region signals to voxel timeseries signals.

        Parameters
        ----------
        imgs: a Niimg-like images/objects
            Data on which regions signals are transformed to voxel
            time series signals.

        Returns
        -------
        region_signals_: a numpy array
            a list of averaged timeseries signals of each of the region.
        """
        signals = []
        regions_ = check_niimg_4d(self.regions_)
        nifti_maps_masker = NiftiMapsMasker(regions_, self.masker_.mask_img_)
        nifti_maps_masker.fit()

        for img in imgs:
            region_signals = nifti_maps_masker.transform(img)
            signals.append(region_signals)

        self.region_signals_ = signals

        return self

