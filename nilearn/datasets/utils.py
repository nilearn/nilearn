"""Downloading NeuroImaging datasets: utility functions."""

import numbers
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from nibabel.spatialimages import SpatialImage
from scipy.ndimage import binary_fill_holes

from nilearn._utils.docs import fill_doc
from nilearn._utils.ndimage import get_border_data
from nilearn._utils.niimg import safe_get_data
from nilearn._utils.param_validation import check_params
from nilearn.datasets import load_mni152_template
from nilearn.image import check_niimg_3d, get_data, new_img_like
from nilearn.image.resampling import reorder_img


@fill_doc
def get_data_dirs(data_dir=None):
    """Return the directories in which nilearn looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    Parameters
    ----------
    %(data_dir)s

    Returns
    -------
    paths : list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :

    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NILEARN_SHARED_DATA
    4. the user environment variable NILEARN_DATA
    5. nilearn_data in the user home folder

    """
    check_params(locals())

    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(str(data_dir).split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv("NILEARN_SHARED_DATA")
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv("NILEARN_DATA")
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        paths.append(str(Path("~/nilearn_data").expanduser()))
    return paths


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
            anat_img = reorder_img(anat_img)
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


def _load_mni152_template(
    anat_img, black_bg: Literal["auto"] | bool
) -> tuple[Any, bool, float, float]:
    """Load MNI152 template.

    Parameters
    ----------
    anat_img :
        Template to load
    black_bg : Literal["auto"] | bool
        Whether to use a black background. If "auto", it will be set to False
        for the MNI152 template.

    Returns
    -------
    tuple[Any, bool, float, float]
        The loaded template, whether to use a black background, and the vmin
        and vmax values for the template.
    """
    anat_img.load()
    # We special-case the 'canonical anat', as we don't need
    # to do a few transforms to it.
    vmin = 0
    vmax = anat_img.vmax
    if black_bg == "auto":
        black_bg = False
    return anat_img, black_bg, vmin, vmax


def _load_custom_anat(
    anat_img, dim: Literal["auto"] | float, black_bg: Literal["auto"] | bool
) -> tuple[Any, bool, float | None, float | None]:
    """Load custom anatomy image.

    Compute vmin/vmax and black_bg, if needed.

    Parameters
    ----------
    anat_img : _type_
        The anatomy image to load.
    dim : Literal["auto"] | float
        The dimming factor.
    black_bg : Literal["auto"] | bool
        Whether to use a black background. If "auto", it will be set based on
        the values of the image.

    Returns
    -------
    tuple[Any, bool, float | None, float | None]
        The loaded anatomy image, whether to use a black background, and the
        vmin and vmax values for the image.
    """
    anat_img = check_niimg_3d(anat_img)
    # Clean anat_img for non-finite values to avoid computing unnecessary
    # border data values.
    data = safe_get_data(anat_img, ensure_finite=True)
    anat_img = new_img_like(anat_img, data, affine=anat_img.affine)

    vmin: float | None = None
    vmax: float | None = None
    if dim or (black_bg == "auto"):
        # We need to inspect the values of the image
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))

        if black_bg == "auto":
            # Guess if the background is rather black or light based on
            # the values of voxels near the border
            background = np.median(get_border_data(data, 2))
            black_bg = bool(background <= 0.5 * (vmin + vmax))

    return anat_img, black_bg, vmin, vmax


def _apply_dimming(
    dim: Literal["auto"] | float, black_bg: bool, vmin: float, vmax: float
) -> tuple[float, float]:
    """Apply dimming to vmin/vmax.

    Parameters
    ----------
    dim : Literal["auto"] | float
        Dimming factor. If "auto", it will be set to 0.8 for black background
        and 0.6 for light background.
    black_bg : bool
        Whether the background is black or light.
    vmin : float
        Minimum value of the image data.
    vmax : float
        Maximum value of the image data.

    Returns
    -------
    tuple[float, float]
        The new vmin and vmax values after applying dimming.

    Raises
    ------
    ValueError
        If dim is not "auto" nor a number.
    """
    if dim != "auto" and not isinstance(dim, numbers.Number):
        raise ValueError(
            "The input given for 'dim' needs to be a float. "
            f"You provided dim={dim} in {dim.__class__.__name__}."
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

    return vmin, vmax


# The constant that we use as a default in functions
MNI152TEMPLATE = _MNI152Template()


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
        anat_img, black_bg, vmin, vmax = _load_mni152_template(
            anat_img, black_bg
        )
    else:
        anat_img, black_bg, vmin, vmax = _load_custom_anat(
            anat_img, dim, black_bg
        )

    if dim:
        vmin, vmax = _apply_dimming(dim, black_bg, vmin, vmax)

    return anat_img, black_bg, vmin, vmax
