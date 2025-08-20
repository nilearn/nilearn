"""
Plotting utilities
that can be used outside of the plotting.image subpackage.
"""

import numbers

import numpy as np
from nibabel.spatialimages import SpatialImage
from scipy.ndimage import binary_fill_holes

from nilearn._utils.ndimage import get_border_data
from nilearn._utils.niimg import safe_get_data
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn.datasets import load_mni152_template
from nilearn.image import get_data, new_img_like
from nilearn.image.resampling import reorder_img
from nilearn.plotting._utils import get_cbar_ticks


# A constant class to serve as a sentinel for the default MNI template
class _MNI152Template(SpatialImage):
    """Constant pointing to the MNI152 Template provided by nilearn."""

    data = None
    _affine = None
    vmax = None
    _shape = None
    # Having a header is required by the load_niimg function
    header = None  # type: ignore[assignment]

    def __init__(self, data=None, affine=None, header=None):
        # Comply with spatial image requirements while allowing empty init
        pass

    def load(self):
        if self.data is None:
            anat_img = load_mni152_template(resolution=2)
            anat_img = reorder_img(anat_img, copy_header=True)
            data = get_data(anat_img)
            data = data.astype(np.float64)
            anat_mask = binary_fill_holes(data > np.finfo(float).eps)
            data = np.ma.masked_array(data, np.logical_not(anat_mask))
            self._affine = anat_img.affine
            self.data = data
            self.vmax = data.max()
            self._shape = anat_img.shape

    @property
    def _data_cache(self):
        self.load()
        return self.data

    @property
    def _dataobj(self):
        self.load()
        return self.data

    def get_data(self):
        self.load()
        return self.data

    @property
    def affine(self):
        self.load()
        return self._affine

    def get_affine(self):
        self.load()
        return self._affine

    @property
    def shape(self):
        self.load()
        return self._shape

    def get_shape(self):
        self.load()
        return self._shape

    def __str__(self):
        return "<MNI152Template>"

    def __repr__(self):
        return "<MNI152Template>"


# The constant that we use as a default in functions
MNI152TEMPLATE = _MNI152Template()


def get_cropped_cbar_ticks(cbar_vmin, cbar_vmax, threshold=None, n_ticks=5):
    """Return ticks for cropped colorbars."""
    new_tick_locs = np.linspace(cbar_vmin, cbar_vmax, n_ticks)
    if threshold is not None:
        # Case where cbar is either all positive or all negative
        if 0 <= cbar_vmin <= cbar_vmax or cbar_vmin <= cbar_vmax <= 0:
            idx_closest = np.argmin(
                [abs(abs(new_tick_locs) - threshold) for _ in new_tick_locs]
            )
            new_tick_locs[idx_closest] = threshold
        # Case where we do a symmetric thresholding
        # within an asymmetric cbar
        # and both threshold values are within bounds
        elif cbar_vmin <= -threshold <= threshold <= cbar_vmax:
            new_tick_locs = get_cbar_ticks(
                cbar_vmin, cbar_vmax, threshold, n_ticks=len(new_tick_locs)
            )
        # Case where one of the threshold values is out of bounds
        else:
            idx_closest = np.argmin(
                [abs(new_tick_locs - threshold) for _ in new_tick_locs]
            )
            new_tick_locs[idx_closest] = (
                -threshold if threshold > cbar_vmax else threshold
            )
    return new_tick_locs


def load_anat(anat_img=MNI152TEMPLATE, dim="auto", black_bg="auto"):
    """Load anatomy, for optional diming."""
    vmin = None
    vmax = None
    if anat_img is False or anat_img is None:
        if black_bg == "auto":
            # No anatomy given: no need to turn black_bg on
            black_bg = False
        return anat_img, black_bg, vmin, vmax

    if anat_img is MNI152TEMPLATE:
        anat_img.load()
        # We special-case the 'canonical anat', as we don't need
        # to do a few transforms to it.
        vmin = 0
        vmax = anat_img.vmax
        if black_bg == "auto":
            black_bg = False
    else:
        anat_img = check_niimg_3d(anat_img)
        # Clean anat_img for non-finite values to avoid computing unnecessary
        # border data values.
        data = safe_get_data(anat_img, ensure_finite=True)
        anat_img = new_img_like(anat_img, data, affine=anat_img.affine)
        if dim or black_bg == "auto":
            # We need to inspect the values of the image
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        if black_bg == "auto":
            # Guess if the background is rather black or light based on
            # the values of voxels near the border
            background = np.median(get_border_data(data, 2))
            black_bg = not (background > 0.5 * (vmin + vmax))
    if dim:
        if dim != "auto" and not isinstance(dim, numbers.Number):
            raise ValueError(
                "The input given for 'dim' needs to be a float. "
                f"You provided dim={dim} in {type(dim)}."
            )
        vmean = 0.5 * (vmin + vmax)
        ptp = 0.5 * (vmax - vmin)
        if black_bg:
            if not isinstance(dim, numbers.Number):
                dim = 0.8
            vmax = vmean + (1 + dim) * ptp
        else:
            if not isinstance(dim, numbers.Number):
                dim = 0.6
            vmin = 0.5 * (2 - dim) * vmean - (1 + dim) * ptp
    return anat_img, black_bg, vmin, vmax
