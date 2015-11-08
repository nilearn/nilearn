"""
Better brain parcellations for Region of Interest analysis
"""
import numbers
import nibabel
import numpy as np

from scipy.ndimage import label

from sklearn.base import clone
from sklearn.externals.joblib import Memory

from .. import masking
from ..input_data import NiftiMapsMasker
from .._utils import check_niimg, check_niimg_4d
from ..image import new_img_like, resample_img
from ..image.image import _smooth_array, threshold_img
from .._utils.niimg_conversions import concat_niimgs, _check_same_fov
from .._utils.compat import _basestring
from ..externals.skimage import peak_local_max, random_walker


def connected_regions(maps_img, min_size=50,
                      extract_type='local_regions',
                      peak_local_smooth=6, mask_img=None):
    """ A function takes the connected components of the brain activation
    maps/regions and breaks each component into a seperate brain regions.

    Parameters
    ----------
    maps_img: a Nifti-like image/object
        an image of the activation or atlas maps which should be extracted
        into a set of regions.

    min_size: int, default is 50
        An integer which denotes the size of voxels in the each region.
        Only the size of the regions which are more than this number
        are kept.

    extract_type: string {"connected_components", "local_regions"} \
        default is local_regions
        A method used to segment/seperate the regions.
        If 'connected_components', each component/region in the image is extracted
        automatically by assigning unique labels based upon their features.
        If 'local_regions', each component/region in the image is extracted based upon their
        local peak max value giving each region a unique identity and then using random walker
        to increase the robustness in the regions seperation.

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

    index_of_each_map: a numpy array
        an array of list of indices where each index value denotes the indentity
        of each extracted region belonging to the family of brain maps.
    """
    all_regions_imgs = []
    index_of_each_map = []
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
            mask_data, _ = masking._load_mask_img(mask_img)
            # Set as 0 to the values which are outside of the mask
            maps[mask_data == 0.] = 0.

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
            try:
                rw_maps = random_walker(map_, seeds_label, mode='cg_mg')
            except Exception as e:
                print("Random Walker algorithm failed for mode:%s, %s" % ('cg_mg', str(e)))
                rw_maps = random_walker(map_, seeds_label, mode='bf')

            # Now simply replace "-1" with "0" for regions seperation
            rw_maps[rw_maps == -1] = 0.
            label_maps = rw_maps
        else:
            # Taking directly the input data which is the case in 'auto' type
            maps_assigned = map_
            # Connected component extraction
            label_maps, n_labels = label(maps_assigned)

        # Takes the size of each labelized region data
        labels_size = np.bincount(label_maps.ravel())
        labels_size[0] = 0.
        for label_id, label_size in enumerate(labels_size):
            if label_size > min_size:
                region_data = (label_maps == label_id) * map_
                region_img = new_img_like(maps_img, region_data)
                regions.append(region_img)

        index_of_each_map.extend([index] * len(regions))
        all_regions_imgs.extend(regions)

    regions_extracted = concat_niimgs(all_regions_imgs)

    return regions_extracted, index_of_each_map


class RegionExtractor(NiftiMapsMasker):
    """ Class to extract connected regions into seperate regions.

    Region Extraction is a post processing technique which
    is implemented to automatically segment each brain atlas maps
    into different set of seperated brain activated region.
    Particularly, to show that each decomposed brain maps can be
    used to focus on a target specific Regions of Interest analysis.

    Parameters
    ----------
    maps_img: 4D Niimg-like object or path to the 4D Niimg
       an image or a filename of the image which contains a set of whole brain
       atlas maps or statistically decomposed brain maps.

    mask_img: Niimg-like object or None,  default is None, optional
        Mask to be applied on the input data.
        If given, mask_img is submitted to an instance of NiftiMapsMasker
        for a preparation towards input maps data.
        If None, no mask will be applied on the data.

    min_size: int, default size 50, optional
        An integer which suppress the smallest spurious regions based upon
        the size of voxels. For example, min_size=50 means that regions
        which have more than cluster of 50 voxels are kept.

    threshold: a float or a string or a number, default is string "80%", optional
        A given input is used to threshold the input maps.
        If given as float, this intensity value will be directly used to threshold
        the maps image. The value should be within the range of minimum intensity and
        maximum intensity of the given input maps.
        Mostly suitable, if user knows exactly which part of the brain regions
        are to be kept. This case is used in thresholding_strategy='img_value'.
        or
        If given as string, it should finish with percent sign e.g. "80%" and
        should be within the range of "0%" to "100%".
        if given as number, it should be a real number of range between 0 and 100.
        Both string and real number are in thresholding_strategy='percentile'.

    thresholding_strategy: string {'percentile', 'img_value'}, default 'percentile', optional
        A strategy which takes the value and thresholds the maps image.
        Each strategy takes the given threshold into account and applies on the image.
        If 'percentile', images are thresholded based on the percentage of the score on the
        data and the scores which are survived above this percentile are kept.
        or
        If 'img_value', voxels which have intensities greater than the float value
        are kept.

    extractor: string {'connected_components', 'local_regions'} default 'local_regions', optional
        A string as a method used for regions extraction.
        If 'connected_components', each component/region in the image is extracted
        automatically by assigning unique labels based upon their features.
        If 'local_regions', each component/region in the image is extracted based upon their
        local peak max value giving each region a unique identity and then using random walker
        to increase the robustness in the regions seperation.

    peak_local_smooth: scalar, default 6mm, optional
        a smooth parameter value in mm used to smooth an image before locating
        the peak local max value.

    standardize: boolean, True or False, default False, optional
        If True, the time series signals are centered and normalized by
        putting their mean to 0 and variance to 1. Recommended to
        set as True if signals are not already standardized.
        passed to class NiftiMapsMasker.

    detrend: boolean, True or False, default False, optional
        This parameter is passed to nilearn.signal.clean basically
        indicates whether to detrend timeseries signals or not.
        passed to class NiftiMapsMasker.

    low_pass: float, default is None, optional
        This value will be applied on the signals by passing to signal.clean
        Please see the related documentation signal.clean for more details.
        passed to class NiftiMapsMasker.

    high_pass: float, default is None, optional
        This value will be applied on the signals by passing to signal.clean
        Please see the related documentation signal.clean for more details.
        passed to NiftiMapsMasker.

    t_r: float, default is None, optional
        Repetition time in sec. This value is given to signal.clean
        Please see the related documentation for details.
        passed to NiftiMapsMasker.

    memory: instance of joblib.Memory, string, default is None, optional
        Used to cache the masking process. If a string is given, the path
        is set with this string as a folder name in the directory.
        passed to NiftiMapsMasker.

    memory_level: int, default is 0, optional
        Aggressiveness of memory catching. The higher the number, the higher
        the number of functions that will be cached. Zero mean no caching.
        passed to NiftiMapsMasker.

    verbose: int, default is 0, optional
        Indicates the level of verbosity by printing the message. Zero indicates
        nothing is printed.

    Attributes
    ----------
    regions_: Niimg-like image/object
        a list of seperated regions with each region lying on a 3D volume
        concatenated into a 4D Nifti like object.

    index_: a numpy array
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
    def __init__(self, maps_img, mask_img=None, min_size=50,
                 threshold="80%", thresholding_strategy='percentile',
                 extractor='local_regions', peak_local_smooth=6.,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0, verbose=0):
        super(RegionExtractor, self).__init__(
            maps_img=maps_img, mask_img=mask_img,
            standardize=standardize, detrend=detrend, low_pass=low_pass,
            high_pass=high_pass, t_r=t_r, memory=memory,
            memory_level=memory_level, verbose=verbose)
        self.maps_img = maps_img
        self.min_size = min_size
        self.thresholding_strategy = thresholding_strategy
        self.threshold = threshold
        self.extractor = extractor
        self.peak_local_smooth = peak_local_smooth

    def fit(self, X=None, y=None):
        """ Prepare the data and setup for the region extraction

        """
        maps_img = check_niimg_4d(self.maps_img)

        # foreground extraction
        self.threshold_maps_img_ = threshold_img(
            self.maps_img,
            mask_img=self.mask_img,
            threshold=self.threshold,
            thresholding_strategy=self.thresholding_strategy)

        # connected component extraction
        self.regions_, self.index_ = connected_regions(
            self.threshold_maps_img_,
            self.min_size,
            self.extractor,
            self.peak_local_smooth)

        self.maps_img = self.regions_
        super(RegionExtractor, self).fit()

        return self
